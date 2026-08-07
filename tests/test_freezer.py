from pathlib import Path
import os
import platform
import plistlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from py2bin.freezer import (
    _drop_excluded,
    _frozen_macos_app,
    _shell_launcher,
    drop_debug_symbols,
    drop_unused_libraries,
    extract_wheel,
    zip_bytecode,
)
from py2bin.native.launcher import macos_shell_launcher
from py2bin.onefile import _powershell_script, create_onefile


class FreezerTests(unittest.TestCase):
    def test_windows_onefile_script_uses_launcher_environment_without_wmi(self):
        script = _powershell_script(
            offset=1234,
            digest="0" * 64,
            launcher="Demo.exe",
        )
        self.assertIn("$env:PY2BIN_ONEFILE_SELF", script)
        self.assertIn("$env:PY2BIN_ONEFILE_COMMAND", script)
        self.assertNotIn("Get-CimInstance", script)
        self.assertNotIn("Win32_Process", script)
        self.assertNotIn("Join-Path", script)
        self.assertNotIn("Test-Path", script)
        self.assertNotIn("Remove-Item", script)
        self.assertNotIn("Move-Item", script)
        self.assertNotIn("New-Object", script)
        self.assertEqual(
            script.count("if(![IO.File]::Exists($m))"),
            2,
        )
        self.assertLess(
            script.index("if(![IO.File]::Exists($m))"),
            script.index("[Threading.Mutex]::new"),
        )
        self.assertIn(
            "$si=[Diagnostics.ProcessStartInfo]::new()",
            script,
        )

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "self-extracting Mach-O runs only on Apple Silicon",
    )
    def test_onefile_macho_extracts_once_and_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            inner = payload / "inner.bin"
            inner.write_text(
                '#!/bin/sh\nprintf "onefile:%s" "$1"\n',
                encoding="utf-8",
            )
            inner.chmod(0o755)
            output = root / "Demo.bin"
            result = create_onefile(
                payload,
                output,
                target="darwin-arm64",
                launcher=inner,
            )
            environment = os.environ.copy()
            environment["PY2BIN_CACHE_DIR"] = str(root / "cache")
            first = subprocess.run(
                [str(output), "forwarded"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            second = subprocess.run(
                [str(output), "cached"],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(first.stdout, "onefile:forwarded")
            self.assertEqual(second.stdout, "onefile:cached")
            self.assertEqual(output.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            self.assertGreater(result.archive_bytes, 0)
            self.assertEqual(
                len(list((root / "cache").rglob(".py2bin-complete"))),
                1,
            )

    def test_native_macos_launcher_is_a_macho(self):
        image = macos_shell_launcher("exit 0", machine="arm64")
        self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")

    def test_x86_64_macos_launcher_reads_the_initial_stack(self):
        # The x86-64 Mach-O writer uses LC_UNIXTHREAD, which starts execution
        # at the raw entry point with argc/argv on the initial process stack.
        # Only the arm64 image uses LC_MAIN, where they arrive in registers.
        # Reading rdi/rsi/rdx here would read uninitialised registers, and the
        # launcher would exit 64 instead of running the program.
        image = macos_shell_launcher("exit 0", machine="x86_64")
        prologue = (
            b"\x49\x89\xe5"  # mov r13, rsp
            b"\x4d\x8b\x65\x00"  # mov r12, [r13]  (argc)
            b"\x4d\x8d\x75\x08"  # lea r14, [r13+8] (argv)
        )
        self.assertIn(prologue, image)
        # The LC_MAIN register convention must not be used for this target.
        self.assertNotIn(b"\x49\x89\xfc\x49\x89\xf6\x49\x89\xd7", image)

    def test_x86_64_app_launcher_is_emitted_unsigned(self):
        # arm64 macOS requires a code signature, so that launcher embeds an
        # ad-hoc one sealing Info.plist/CodeResources. Intel macOS still loads
        # unsigned executables, so the same request must produce a valid
        # x86-64 Mach-O rather than being refused.
        image = macos_shell_launcher(
            "exit 0",
            machine="x86_64",
            info_plist=b"<plist/>",
            code_resources=b"<plist/>",
        )
        self.assertEqual(image[:4], b"\xcf\xfa\xed\xfe")
        # cputype in the Mach-O header: CPU_TYPE_X86_64 is 0x01000007.
        self.assertEqual(
            int.from_bytes(image[4:8], "little"), 0x01000007
        )

    @unittest.skipUnless(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "native launcher execution requires Apple Silicon",
    )
    def test_native_macos_launcher_forwards_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "launcher"
            launcher.write_bytes(macos_shell_launcher("printf '%s' \"$1\""))
            launcher.chmod(0o755)
            run = subprocess.run(
                [str(launcher), "forwarded"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.stdout, "forwarded")

    def test_posix_launcher_has_no_path_dependent_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "App.bin"
            _shell_launcher(launcher, Path("runtime/bin/python3"), {"PYTHONHOME": "runtime"})
            text = launcher.read_text(encoding="utf-8")
            self.assertNotIn("dirname", text)
            self.assertNotIn("/usr/bin/env", text)
            self.assertIn('exec "$ROOT/runtime/bin/python3"', text)

    def test_extracts_wheel_packages_data_native_files_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "demo-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("demo/__init__.py", "value = 42\n")
                archive.writestr("demo/data/model.json", "{}")
                archive.writestr("demo/native.pyd", b"native")
                archive.writestr("demo-1.0.dist-info/METADATA", "Name: demo\n")
                archive.writestr("demo-1.0.data/purelib/plugin.py", "enabled=True\n")
                archive.writestr("../escape", "bad")
            destination = root / "packages"
            destination.mkdir()
            count = extract_wheel(wheel, destination)
            self.assertEqual(count, 5)
            self.assertTrue((destination / "demo" / "data" / "model.json").exists())
            self.assertTrue((destination / "demo" / "native.pyd").exists())
            self.assertTrue((destination / "demo-1.0.dist-info" / "METADATA").exists())
            self.assertTrue((destination / "plugin.py").exists())
            self.assertFalse((root / "escape").exists())

    def test_compact_wheel_keeps_runtime_payload_and_omits_tests_and_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "demo-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("demo/__init__.py", "value = 42\n")
                archive.writestr("demo/data/schema.json", "{}")
                archive.writestr("demo/native.pyd", b"native")
                archive.writestr("demo/compiled_only.pyc", b"runtime")
                archive.writestr("demo/tests/test_api.py", "assert True\n")
                archive.writestr(
                    "demo/__pycache__/module.pyc",
                    b"bytecode",
                )
                archive.writestr(
                    "demo-1.0.dist-info/METADATA",
                    "Name: demo\nVersion: 1.0\n",
                )
            destination = root / "packages"
            destination.mkdir()
            count = extract_wheel(wheel, destination, compact=True)
            self.assertEqual(count, 5)
            self.assertTrue((destination / "demo" / "__init__.py").is_file())
            self.assertTrue(
                (destination / "demo" / "data" / "schema.json").is_file()
            )
            self.assertTrue((destination / "demo" / "native.pyd").is_file())
            self.assertTrue(
                (destination / "demo" / "compiled_only.pyc").is_file()
            )
            self.assertTrue(
                (
                    destination
                    / "demo-1.0.dist-info"
                    / "METADATA"
                ).is_file()
            )
            self.assertFalse((destination / "demo" / "tests").exists())
            self.assertFalse(
                (destination / "demo" / "__pycache__").exists()
            )

    def test_frozen_macos_app_wraps_payload_and_icon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "payload"
            payload.mkdir()
            payload_launcher = payload / "ManimStudio.bin"
            payload_launcher.write_text("#!/bin/sh\n", encoding="utf-8")
            png = (
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 128, 128)
                + b"\x08\x06\x00\x00\x00"
            )
            icon = root / "icon.ico"
            icon.write_bytes(
                struct.pack("<HHH", 0, 1, 1)
                + struct.pack("<BBBBHHII", 128, 128, 0, 0, 1, 32, len(png), 22)
                + png
            )
            app = root / "ManimStudio.app"
            with mock.patch(
                "py2bin.native.launcher.platform.machine", return_value="arm64"
            ):
                launcher = _frozen_macos_app(
                    payload,
                    app,
                    "ManimStudio",
                    payload_launcher,
                    icon,
                    Path("runtime/bin/python3"),
                    {"PYTHONHOME": "runtime"},
                    "darwin-arm64",
                )
            self.assertTrue(launcher.is_file())
            self.assertTrue(
                (app / "Contents" / "Resources" / "bundle" / "ManimStudio.bin").is_file()
            )
            self.assertEqual(
                (app / "Contents" / "Resources" / "AppIcon.icns").read_bytes()[:4],
                b"icns",
            )
            self.assertTrue(
                (app / "Contents" / "_CodeSignature" / "CodeResources").is_file()
            )
            with (app / "Contents" / "Info.plist").open("rb") as stream:
                info = plistlib.load(stream)
            self.assertEqual(info["CFBundleIconFile"], "AppIcon.icns")


if __name__ == "__main__":
    unittest.main()


class ZipBytecodeTests(unittest.TestCase):
    """The carried library, packed into the archive the interpreter expects."""

    def _bundle(self, root: Path) -> Path:
        """A bundle shaped like the real thing, with one native package."""

        bundle = root / "App.app"
        library = bundle / "Contents" / "lib" / "python3.14"
        (library / "json").mkdir(parents=True)
        (library / "lib-dynload").mkdir(parents=True)
        (library / "native" / "__pycache__").mkdir(parents=True)
        (library / "os.pyc").write_bytes(b"os bytecode" * 40)
        (library / "json" / "__init__.pyc").write_bytes(b"json bytecode" * 40)
        (library / "lib-dynload" / "select.so").write_bytes(b"\xcf\xfa\xed\xfe" * 10)
        # A package holding an extension: dyld needs a file, so it stays put.
        (library / "native" / "ext.so").write_bytes(b"\xcf\xfa\xed\xfe" * 10)
        (library / "native" / "__pycache__" / "helper.cpython-314.pyc").write_bytes(
            b"helper" * 40
        )
        return bundle

    def test_the_library_moves_into_the_archive_the_interpreter_looks_for(self):
        # `{prefix}/lib/pythonXY.zip` is on sys.path whether or not it exists,
        # so no path setup is needed - the name has to be exactly that.
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle(Path(directory))
            zip_bytecode(bundle)
            archive = bundle / "Contents" / "lib" / "python314.zip"
            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as packed:
                names = set(packed.namelist())
        self.assertIn("os.pyc", names)
        self.assertIn("json/__init__.pyc", names)

    def test_a_package_holding_an_extension_is_left_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle(Path(directory))
            zip_bytecode(bundle)
            library = bundle / "Contents" / "lib" / "python3.14"
            with zipfile.ZipFile(library.parent / "python314.zip") as packed:
                names = set(packed.namelist())
            self.assertNotIn("native/helper.pyc", names)
            self.assertTrue(
                (library / "native" / "__pycache__" /
                 "helper.cpython-314.pyc").is_file()
            )
            self.assertTrue((library / "native" / "ext.so").is_file())

    def test_the_name_in_the_archive_is_the_one_import_asks_for(self):
        # `__pycache__/helper.cpython-314.pyc` is imported as `helper`, so a
        # cache directory in the archive would put every module out of reach.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "App.app"
            library = bundle / "Contents" / "lib" / "python3.14" / "pkg"
            (library / "__pycache__").mkdir(parents=True)
            (library / "__pycache__" / "part.cpython-314.pyc").write_bytes(b"x" * 90)
            zip_bytecode(bundle)
            with zipfile.ZipFile(
                bundle / "Contents" / "lib" / "python314.zip"
            ) as packed:
                self.assertEqual(packed.namelist(), ["pkg/part.pyc"])

    def test_storing_is_offered_for_a_filesystem_that_compresses_already(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._bundle(Path(directory))
            zip_bytecode(bundle, compress=False)
            with zipfile.ZipFile(
                bundle / "Contents" / "lib" / "python314.zip"
            ) as packed:
                methods = {item.compress_type for item in packed.infolist()}
        self.assertEqual(methods, {zipfile.ZIP_STORED})

    def test_a_bundle_with_no_carried_library_is_left_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "App.app"
            (bundle / "Contents" / "MacOS").mkdir(parents=True)
            self.assertEqual(zip_bytecode(bundle), 0)
            self.assertEqual(list(bundle.rglob("*.zip")), [])


class DebugSymbolTests(unittest.TestCase):
    """The DWARF companions some wheels ship, which nothing loads."""

    def test_a_debug_companion_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "App.app"
            packages = bundle / "Contents" / "Resources" / "site-packages"
            dwarf = (
                packages / "objc" / "_objc.so.dSYM" / "Contents" / "Resources"
                / "DWARF"
            )
            dwarf.mkdir(parents=True)
            (dwarf / "_objc.so").write_bytes(b"\xcf\xfa\xed\xfe" + b"d" * 5000)
            (packages / "objc" / "_objc.so").write_bytes(b"\xcf\xfa\xed\xfe" * 8)
            freed = drop_debug_symbols(bundle)
        self.assertGreater(freed, 5000)

    def test_the_extension_itself_is_left_alone(self):
        # The companion holds a copy of the binary's name; deleting by name
        # rather than by the `.dSYM` directory would take the real one too.
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "App.app"
            packages = bundle / "Contents" / "Resources" / "site-packages"
            (packages / "objc" / "_objc.so.dSYM").mkdir(parents=True)
            (packages / "objc" / "_objc.so.dSYM" / "x").write_bytes(b"d" * 100)
            extension = packages / "objc" / "_objc.so"
            extension.write_bytes(b"\xcf\xfa\xed\xfe" * 8)
            drop_debug_symbols(bundle)
            self.assertTrue(extension.is_file())
            self.assertFalse((packages / "objc" / "_objc.so.dSYM").exists())

    def test_a_bundle_without_any_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "App.app"
            (bundle / "Contents" / "MacOS").mkdir(parents=True)
            self.assertEqual(drop_debug_symbols(bundle), 0)


class VendoredLibraryTests(unittest.TestCase):
    """A wheel's private `.dylibs`, closed over on its own.

    A wheel with native dependencies ships them beside its extension rather
    than in the bundle's library directory - Pillow puts nineteen in
    `PIL/.dylibs`. They answer to the extensions of *their* package, so the
    closure has to be computed there rather than over the bundle at large,
    which is what the first version of this missed: it looked only in
    `Contents/lib`, so excluding a codec removed the extension and left its
    2.9 MB library sitting beside it.
    """

    def _package(self, root: Path) -> Path:
        bundle = root / "App.app"
        pil = bundle / "Contents" / "Resources" / "site-packages" / "PIL"
        (pil / ".dylibs").mkdir(parents=True)
        (pil / ".dylibs" / "libjpeg.62.dylib").write_bytes(b"j" * 4000)
        (pil / ".dylibs" / "libavif.16.dylib").write_bytes(b"a" * 9000)
        # The core extension names only the codec it really uses.
        (pil / "_imaging.cpython-314-darwin.so").write_bytes(
            b"\xcf\xfa\xed\xfe" + b"@loader_path/.dylibs/libjpeg.62.dylib\0"
        )
        (pil / "_avif.cpython-314-darwin.so").write_bytes(
            b"\xcf\xfa\xed\xfe" + b"@loader_path/.dylibs/libavif.16.dylib\0"
        )
        return bundle

    def test_a_vendored_library_still_wanted_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._package(Path(directory))
            drop_unused_libraries(bundle)
            dylibs = (
                bundle / "Contents" / "Resources" / "site-packages" / "PIL" / ".dylibs"
            )
            self.assertTrue((dylibs / "libjpeg.62.dylib").is_file())
            self.assertTrue((dylibs / "libavif.16.dylib").is_file())

    def test_a_vendored_library_whose_extension_went_is_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self._package(Path(directory))
            pil = bundle / "Contents" / "Resources" / "site-packages" / "PIL"
            (pil / "_avif.cpython-314-darwin.so").unlink()
            freed = drop_unused_libraries(bundle)
            self.assertFalse((pil / ".dylibs" / "libavif.16.dylib").exists())
            self.assertTrue((pil / ".dylibs" / "libjpeg.62.dylib").is_file())
        self.assertGreaterEqual(freed, 9000)


class ExcludedModuleTests(unittest.TestCase):
    """Reaching inside a package the walk had to keep whole."""

    def _roots(self, root: Path) -> list[Path]:
        packages = root / "site-packages"
        (packages / "PIL").mkdir(parents=True)
        (packages / "PIL" / "AvifImagePlugin.py").write_text("x = 1\n")
        (packages / "PIL" / "Image.py").write_text("y = 2\n")
        (packages / "PIL" / "_avif.cpython-314-darwin.so").write_bytes(b"so" * 50)
        return [packages]

    def test_both_halves_of_a_codec_can_be_named(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = self._roots(root)
            freed = _drop_excluded(roots, ("PIL.AvifImagePlugin", "PIL._avif"))
            pil = root / "site-packages" / "PIL"
            self.assertFalse((pil / "AvifImagePlugin.py").exists())
            self.assertFalse(list(pil.glob("_avif.*.so")))
            # Everything not named is untouched.
            self.assertTrue((pil / "Image.py").is_file())
        self.assertGreater(freed, 0)

    def test_naming_something_absent_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self._roots(Path(directory))
            self.assertEqual(_drop_excluded(roots, ("PIL.NotThere",)), 0)


class FrozenEntryDundersTests(unittest.TestCase):
    """A frozen program's `__main__` is the one the interpreter would make.

    `runpy.run_path` is the obvious way to start the entry and is not quite
    what CPython does with a script named on its command line - it sets
    `__package__` to the empty string where a script has None, and `exec`
    puts the builtins *dictionary* where `__main__` has the module. Both are
    the kind of difference that only shows up inside somebody else's library.
    """

    def _bootstrap(self) -> str:
        """The bootstrap the freezer writes, without running a whole build."""

        import inspect

        from py2bin import freezer

        source = inspect.getsource(freezer)
        start = source.index('(stage / "py2bin_bootstrap.py").write_text(')
        return source[start : source.index("encoding=", start)]

    def test_the_entry_is_run_the_way_a_script_is_run(self):
        written = self._bootstrap()
        # The call, not the prose: the comment beside it explains why it
        # is not used, and says the name.
        self.assertNotIn("runpy.run_path(", written)
        self.assertIn("module.__package__ = None", written)
        self.assertIn("module.__spec__ = None", written)
        self.assertIn("module.__builtins__ = builtins", written)
        self.assertIn("SourceFileLoader", written)
        # Where it is written, and both ways in - the plain launcher and the
        # one that reports a crash to a log - going through the same starter.
        self.assertIn("def _run(entry):", written)
        self.assertEqual(written.count("_run(entry)"), 3)

    def test_the_archive_bootstrap_agrees_with_it(self):
        """`bootstrap.py` starts the onefile shape and had the same bug."""

        root = Path(__file__).resolve().parents[1]
        written = (root / "src" / "py2bin" / "bootstrap.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("runpy.run_path(", written)
        for line in (
            "module.__package__ = None",
            "module.__spec__ = None",
            "module.__builtins__ = builtins",
        ):
            self.assertIn(line, written)


class FrameworkStubTests(unittest.TestCase):
    """Some `bin/python3` is a stub that hands over to another file.

    Homebrew's is: 52 KB, with `Resources/Python.app/Contents/MacOS/Python`
    written inside it. A bundle carrying only `bin/python3` built cleanly,
    reported success, and died at start-up with a `posix_spawn` error naming
    a file it did not have - so the tier billed as "every Python program
    works" did not work at all for anyone whose Python came from Homebrew.
    """

    def test_the_stub_target_is_carried_when_the_stub_names_it(self):
        import inspect

        from py2bin import freezer

        written = inspect.getsource(freezer._freeze_current_runtime)
        # Asked of the executable, not assumed of the distribution: the stub
        # says where it is going, so the test is whether it says so.
        self.assertIn(
            b"Resources/Python.app/Contents/MacOS/Python", written.encode()
        )
        self.assertIn("executable_source.read_bytes()", written)
        self.assertIn("shutil.copytree(source_app", written)

    @unittest.skipUnless(
        platform.system() == "Darwin", "framework layout is macOS only"
    )
    def test_this_python_would_produce_a_runnable_bundle(self):
        """Whatever Python is running the suite, its runtime copies whole.

        Reading the two files rather than building a bundle: a freeze takes
        seconds and this is the part that was wrong.
        """

        import sysconfig

        if not sysconfig.get_config_var("PYTHONFRAMEWORK"):
            self.skipTest("not a framework build")
        source = Path(sys.executable).resolve()
        if not source.is_file():
            self.skipTest("no resolvable executable")
        names_the_app = (
            b"Resources/Python.app/Contents/MacOS/Python" in source.read_bytes()
        )
        if not names_the_app:
            return
        app = Path(sys.base_prefix) / "Resources" / "Python.app"
        self.assertTrue(
            app.is_dir(),
            f"{source} hands over to {app}, which is not there to carry",
        )
