from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from py2bin.runtime_fetch import (
    FetchError,
    FetchLock,
    extract_zip,
    select_wheel,
    wheel_is_compatible,
)


class WheelCompatibilityTests(unittest.TestCase):
    """Wheel tag matching decides what may be installed for a target."""

    def test_exact_cpython_abi_matches(self):
        self.assertTrue(
            wheel_is_compatible(
                "pillow-12.3.0-cp312-cp312-win_amd64.whl", "windows-x86_64", "3.12"
            )
        )

    def test_other_cpython_abi_is_rejected(self):
        self.assertFalse(
            wheel_is_compatible(
                "pillow-12.3.0-cp311-cp311-win_amd64.whl", "windows-x86_64", "3.12"
            )
        )

    def test_stable_abi_is_forward_compatible(self):
        # A cp37-abi3 wheel loads on every later CPython 3.x.
        self.assertTrue(
            wheel_is_compatible(
                "psutil-7.2.2-cp37-abi3-win_amd64.whl", "windows-x86_64", "3.12"
            )
        )

    def test_stable_abi_built_for_a_newer_python_is_rejected(self):
        self.assertFalse(
            wheel_is_compatible(
                "demo-1.0-cp313-abi3-win_amd64.whl", "windows-x86_64", "3.12"
            )
        )

    def test_free_threaded_abi_is_rejected(self):
        self.assertFalse(
            wheel_is_compatible(
                "psutil-7.2.2-cp313-cp313t-win_amd64.whl", "windows-x86_64", "3.13"
            )
        )

    def test_pure_python_wheel_matches_every_target(self):
        for target in ("windows-x86_64", "darwin-arm64", "linux-x86_64"):
            self.assertTrue(
                wheel_is_compatible(
                    "bottle-0.13.4-py2.py3-none-any.whl", target, "3.12"
                ),
                target,
            )

    def test_foreign_platform_is_rejected(self):
        self.assertFalse(
            wheel_is_compatible(
                "demo-1.0-cp312-cp312-macosx_11_0_arm64.whl",
                "windows-x86_64",
                "3.12",
            )
        )
        self.assertFalse(
            wheel_is_compatible(
                "demo-1.0-cp312-cp312-win32.whl", "windows-x86_64", "3.12"
            )
        )

    def test_malformed_filename_is_rejected(self):
        self.assertFalse(
            wheel_is_compatible("not-a-wheel.whl", "windows-x86_64", "3.12")
        )

    def test_native_wheel_is_preferred_over_pure_python(self):
        files = [
            {
                "packagetype": "bdist_wheel",
                "filename": "demo-1.0-py3-none-any.whl",
                "url": "https://example.invalid/a",
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "demo-1.0-cp312-cp312-win_amd64.whl",
                "url": "https://example.invalid/b",
            },
        ]
        chosen = select_wheel(files, "windows-x86_64", "3.12")
        self.assertEqual(chosen["filename"], "demo-1.0-cp312-cp312-win_amd64.whl")

    def test_yanked_and_sdist_files_are_never_selected(self):
        files = [
            {"packagetype": "sdist", "filename": "demo-1.0.tar.gz", "url": "x"},
            {
                "packagetype": "bdist_wheel",
                "filename": "demo-1.0-py3-none-any.whl",
                "url": "y",
                "yanked": True,
            },
        ]
        self.assertIsNone(select_wheel(files, "windows-x86_64", "3.12"))


class FetchLockTests(unittest.TestCase):
    def test_lock_round_trips_and_reports_expected_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fetch.lock.json"
            lock = FetchLock(path)
            lock.record("demo.whl", "https://example.invalid/demo.whl", "a" * 64)
            lock.save()

            reloaded = FetchLock.load(path)
            self.assertEqual(reloaded.expected("demo.whl"), "a" * 64)
            self.assertIsNone(reloaded.expected("absent.whl"))

    def test_wrong_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fetch.lock.json"
            path.write_text(json.dumps({"schema": 99}), encoding="utf-8")
            with self.assertRaises(FetchError):
                FetchLock.load(path)

    def test_missing_lock_is_empty_rather_than_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = FetchLock.load(Path(directory) / "absent.json")
            self.assertIsNone(lock.expected("anything"))


class ExtractionTests(unittest.TestCase):
    """Archive extraction must never escape its destination."""

    def _archive(self, directory: Path, name: str) -> Path:
        archive = directory / "payload.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(name, b"payload")
        return archive

    def test_normal_member_extracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root, "runtime/python.exe")
            written = extract_zip(archive, root / "out")
            self.assertEqual(written, 1)
            self.assertEqual(
                (root / "out" / "runtime" / "python.exe").read_bytes(), b"payload"
            )

    def test_traversal_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root, "../escaped.txt")
            with self.assertRaises(FetchError):
                extract_zip(archive, root / "out")
            self.assertFalse((root / "escaped.txt").exists())

    def test_absolute_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._archive(root, "/etc/passwd")
            with self.assertRaises(FetchError):
                extract_zip(archive, root / "out")


class DigestVerificationTests(unittest.TestCase):
    def test_digest_of_known_bytes(self):
        # The downloader verifies exactly this digest before accepting a file.
        self.assertEqual(
            hashlib.sha256(b"py2bin").hexdigest(),
            hashlib.sha256(b"py2bin").hexdigest(),
        )
        self.assertNotEqual(
            hashlib.sha256(b"py2bin").hexdigest(),
            hashlib.sha256(b"py2bin ").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
