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


class PackedBesideExecutableTests(unittest.TestCase):
    """`--onefile` where there is no bundle: the program and what it carries."""

    def _tree(self, root: Path) -> Path:
        program = root / "prog"
        program.write_bytes(b"#!/bin/sh\necho hi\n")
        program.chmod(0o755)
        (root / "site-packages").mkdir()
        (root / "site-packages" / "mod.py").write_text("x = 1")
        (root / "web").mkdir()
        (root / "web" / "index.html").write_text("<p>hi</p>")
        # A build leaving that must not travel with the program.
        (root / "prog.capi.c").write_text("/* generated */")
        return program

    def test_the_payload_holds_the_program_and_what_it_carries(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            program = self._tree(root)
            held, total = onefile.pack_beside_executable(
                program, "linux-x86_64", ["site-packages", "web"]
            )
            self.assertEqual(held, 3)  # prog, mod.py, index.html
            self.assertGreater(total, 0)
            image = program.read_bytes()
            self.assertIn(b"PY2BIN-ONEFILE-PAYLOAD-V1:", image)

    def test_the_generated_c_is_not_packed(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            program = self._tree(root)
            onefile.pack_beside_executable(
                program, "linux-x86_64", ["site-packages", "web"]
            )
            self.assertNotIn(b"/* generated */", program.read_bytes())

    def test_what_sat_beside_the_program_is_left_alone(self):
        # There is no way from here to tell a directory the build copied in
        # from one the user keeps there, and deleting the second is data loss.
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            program = self._tree(root)
            onefile.pack_beside_executable(
                program, "linux-x86_64", ["site-packages", "web"]
            )
            self.assertTrue((root / "web" / "index.html").is_file())
            self.assertTrue((root / "site-packages" / "mod.py").is_file())

    def test_a_name_that_is_not_there_is_skipped(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            program = self._tree(root)
            held, _ = onefile.pack_beside_executable(
                program, "linux-x86_64", ["site-packages", "web", "absent"]
            )
            self.assertEqual(held, 3)

    def test_a_missing_program_is_refused(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(ValueError):
                onefile.pack_beside_executable(
                    Path(scratch) / "nothing", "linux-x86_64", []
                )


class WindowsChildProcess(unittest.TestCase):
    """The PowerShell stage has to let the program it starts reach a console.

    This is the same mistake as in the launcher stub the stage is started
    from, one level further down. `CreateNoWindow` on a console program denies
    the child the console it writes to, so the program runs correctly, its
    output is thrown away, and it exits 0 having printed nothing - which reads
    from outside like a program that produces no output rather than one whose
    output was discarded.
    """

    def _script(self, windowed: bool) -> str:
        from py2bin.onefile import _powershell_script

        return _powershell_script(
            offset=4096,
            digest="0" * 64,
            launcher="runtime/app.exe",
            windowed=windowed,
        )

    def test_a_console_build_leaves_the_child_a_console(self) -> None:
        self.assertIn("$si.CreateNoWindow=$false;", self._script(windowed=False))

    def test_a_windowed_build_still_suppresses_it(self) -> None:
        self.assertIn("$si.CreateNoWindow=$true;", self._script(windowed=True))

    def test_failures_are_reported_rather_than_serialised(self) -> None:
        # Errors used to go to PowerShell's error stream, which is serialised
        # as CLIXML when redirected: a failure arrived as a page of XML.
        script = self._script(windowed=False)
        self.assertIn("try{", script)
        self.assertIn("}catch{", script)
        self.assertIn("[Console]::Out.WriteLine", script)
        self.assertIn("$ProgressPreference='SilentlyContinue';", script)
