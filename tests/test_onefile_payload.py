"""Packing a directory into one file, without packing the file into itself."""

import tempfile
import unittest
from pathlib import Path

from py2bin import onefile
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


class PackedAppBundleTests(unittest.TestCase):
    """`--onefile` on a macOS `.app`: the payload folds into the executable.

    A macOS application is a directory and has to stay one - Finder runs
    `Contents/MacOS/<name>` and Gatekeeper reads `Contents/Info.plist` beside
    it. So what "one file" means here is that the payload becomes one, not
    that the bundle stops being a bundle.
    """

    def _bundle(self, root: Path) -> Path:
        bundle = root / "Demo.app"
        macos = bundle / "Contents" / "MacOS"
        macos.mkdir(parents=True)
        (bundle / "Contents" / "Info.plist").write_bytes(
            b'<?xml version="1.0"?><plist version="1.0"><dict>'
            b"<key>CFBundleExecutable</key><string>Demo</string>"
            b"</dict></plist>"
        )
        launcher = macos / "Demo"
        launcher.write_bytes(b"#!/bin/sh\necho packed\n")
        launcher.chmod(0o755)
        for n in range(5):
            (bundle / "Contents" / f"data{n}.txt").write_text(f"payload {n}")
        return bundle

    def test_the_bundle_is_reduced_to_its_launcher_and_plist(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            bundle = self._bundle(root)
            before = sum(1 for f in bundle.rglob("*") if f.is_file())
            held, total = onefile.pack_app_bundle(bundle, "darwin-arm64", "Demo")
            self.assertEqual(held, before)
            after = sorted(
                f.relative_to(bundle).as_posix()
                for f in bundle.rglob("*")
                if f.is_file()
            )
            # The launcher, the plist macOS reads beside it, and the manifest
            # the re-seal writes for those two.
            self.assertIn("Contents/MacOS/Demo", after)
            self.assertIn("Contents/Info.plist", after)
            self.assertLess(len(after), before)
            self.assertGreater(total, 0)

    def test_the_launcher_carries_the_payload(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = self._bundle(Path(scratch))
            onefile.pack_app_bundle(bundle, "darwin-arm64", "Demo")
            image = (bundle / "Contents" / "MacOS" / "Demo").read_bytes()
            self.assertIn(b"PY2BIN-ONEFILE-PAYLOAD-V1:", image)
            # Big enough to be the archive rather than just a stub.
            self.assertGreater(len(image), 400)

    def test_a_bundle_without_an_executable_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            bundle = Path(scratch) / "Empty.app"
            (bundle / "Contents" / "MacOS").mkdir(parents=True)
            (bundle / "Contents" / "Info.plist").write_bytes(b"<plist/>")
            with self.assertRaises(ValueError):
                onefile.pack_app_bundle(bundle, "darwin-arm64", "Empty")

    def test_a_directory_that_is_not_a_bundle_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            plain = Path(scratch) / "NotAnApp"
            plain.mkdir()
            with self.assertRaises(ValueError):
                onefile.pack_app_bundle(plain, "darwin-arm64", "NotAnApp")
