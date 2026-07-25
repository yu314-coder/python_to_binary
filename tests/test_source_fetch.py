from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import tempfile
import unittest
import zipfile

from py2bin.cli import main
from py2bin.native import NativeCompileError
from py2bin.source_compile import compile_locked_sources
from py2bin.source_fetch import SOURCE_MANIFEST, fetch_sources_for_entry


class LockedSourceTests(unittest.TestCase):
    def _archive(
        self,
        root: Path,
        source: str = "ANSWER = 42\n",
        member: str = "demo-revision/src/demo/__init__.py",
    ) -> tuple[Path, str]:
        archive = root / "demo-source.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(member, source)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        return archive, digest

    def _lock(self, root: Path, archive: Path, digest: str) -> Path:
        lock = root / "py2bin-sources.lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "sources": {
                        "demo": {
                            "path": archive.name,
                            "revision": "revision",
                            "sha256": digest,
                            "subdirectory": "src",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return lock

    def _application(self, root: Path, source: str) -> tuple[Path, Path]:
        application = root / "application"
        application.mkdir()
        entry = application / "main.py"
        entry.write_text(source, encoding="utf-8")
        return application, entry

    def test_fetches_pinned_archive_and_compiles_imported_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(root)
            lock = self._lock(root, archive, digest)
            application, entry = self._application(
                root,
                "from demo import ANSWER\nraise SystemExit(ANSWER)\n",
            )
            result = compile_locked_sources(
                entry,
                root / "answer",
                source_lock=lock,
                source_cache=root / "cache",
                source_root=application,
                target="darwin-arm64",
            )
            self.assertEqual(result.imports, ("demo",))
            self.assertEqual(result.fetched[0].sha256, digest)
            self.assertTrue(
                (result.fetched[0].root / "demo" / "__init__.py").is_file()
            )
            self.assertTrue((result.fetched[0].root / SOURCE_MANIFEST).is_file())
            self.assertEqual(result.native.artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(result.native.artifact)])
                self.assertEqual(run.returncode, 42)

    def test_cli_fetches_and_reports_locked_source_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(
                root,
                "raise RuntimeError('must not execute during fetch')\nANSWER = 7\n",
            )
            lock = self._lock(root, archive, digest)
            application, entry = self._application(
                root,
                "from demo import ANSWER\nprint(ANSWER)\n",
            )
            output = StringIO()
            with redirect_stdout(output):
                status = main(
                    [
                        "fetch-sources",
                        str(entry),
                        "--source-root",
                        str(application),
                        "--source-lock",
                        str(lock),
                        "--source-cache",
                        str(root / "cache"),
                        "--json",
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["imports"], ["demo"])

    def test_rejects_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, _digest = self._archive(root)
            lock = self._lock(root, archive, "0" * 64)
            application, entry = self._application(root, "import demo\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                fetch_sources_for_entry(
                    entry,
                    application,
                    lock,
                    root / "cache",
                )

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(
                root,
                "bad\n",
                "demo-revision/../../outside.py",
            )
            lock = self._lock(root, archive, digest)
            application, entry = self._application(root, "import demo\n")
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                fetch_sources_for_entry(
                    entry,
                    application,
                    lock,
                    root / "cache",
                )

    def test_rejects_tampered_extracted_source_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(root)
            lock = self._lock(root, archive, digest)
            application, entry = self._application(root, "import demo\n")
            first = fetch_sources_for_entry(
                entry,
                application,
                lock,
                root / "cache",
            )
            (first.roots[0] / "demo" / "__init__.py").write_text(
                "ANSWER = 999\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "tree verification"):
                fetch_sources_for_entry(
                    entry,
                    application,
                    lock,
                    root / "cache",
                )

    def test_downloaded_pure_integer_function_compiles_as_machine_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(
                root,
                "def run():\n"
                "    answer = 6 * 7\n"
                "    return answer\n",
            )
            lock = self._lock(root, archive, digest)
            application, entry = self._application(
                root,
                "from demo import run\nraise SystemExit(run())\n",
            )
            result = compile_locked_sources(
                entry,
                root / "answer",
                source_lock=lock,
                source_cache=root / "cache",
                source_root=application,
                target="darwin-arm64",
            )
            self.assertEqual(result.native.artifact.read_bytes()[:4], b"\xcf\xfa\xed\xfe")
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                run = subprocess.run([str(result.native.artifact)])
                self.assertEqual(run.returncode, 42)

    def test_cli_never_falls_back_to_compatible_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(
                root,
                "def run(value):\n"
                "    return [value]\n",
            )
            lock = self._lock(root, archive, digest)
            application, entry = self._application(
                root,
                "from demo import run\nraise SystemExit(run(1))\n",
            )
            error = StringIO()
            with redirect_stderr(error):
                status = main(
                    [
                        "compile-source",
                        str(entry),
                        "--source-root",
                        str(application),
                        "--source-lock",
                        str(lock),
                        "--source-cache",
                        str(root / "cache"),
                        "--target",
                        "darwin-arm64",
                        "--output",
                        str(root / "bad"),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("signed 64-bit native integer subset", error.getvalue())
            self.assertFalse((root / "bad").exists())

    def test_normal_compile_command_auto_fetches_when_lock_is_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self._archive(root, "ANSWER = 23\n")
            lock = self._lock(root, archive, digest)
            application, entry = self._application(
                root,
                "from demo import ANSWER\nraise SystemExit(ANSWER)\n",
            )
            output = StringIO()
            artifact = root / "answer"
            with redirect_stdout(output):
                status = main(
                    [
                        "compile",
                        str(entry),
                        "--source-root",
                        str(application),
                        "--source-lock",
                        str(lock),
                        "--source-cache",
                        str(root / "cache"),
                        "--target",
                        "darwin-arm64",
                        "--output",
                        str(artifact),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue(artifact.is_file())
            self.assertIn("fetched demo revision revision", output.getvalue())


if __name__ == "__main__":
    unittest.main()
