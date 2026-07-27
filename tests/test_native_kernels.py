"""NumPy/Torch imports must be honestly rejected, never reimplemented.

An earlier experiment reimplemented a static rank-1 integer NumPy/Torch subset
in py2bin IR under an ``--experimental-kernels`` flag. That was removed because
the emitted binary's *observable* result did not match CPython running the real
libraries: a numpy/torch reduction is an ``np.int64`` / 0-d ``Tensor``, not a
plain ``int``, so e.g. ``raise SystemExit(np.sum(...))`` exits 1 under real
NumPy (printing the repr) while the reimplementation exited with the value.
py2bin's absolute-honesty contract forbids shipping such a binary, so these
imports are now rejected with a source-located error for every target and mode.
"""

from pathlib import Path
import tempfile
import unittest

from py2bin.cli import main
from py2bin.native import NativeCompileError, audit_native_library, compile_native, supported_targets


class NumpyTorchRejectionTests(unittest.TestCase):
    def _reject(self, source: str, target: str = "linux-x86_64") -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "prog.py"
            entry.write_text(source, encoding="utf-8")
            with self.assertRaises(NativeCompileError) as caught:
                compile_native(entry, root / "prog", target)
            return str(caught.exception)

    def test_import_numpy_is_rejected_on_every_target(self):
        for target in supported_targets():
            message = self._reject(
                "import numpy as np\n"
                "raise SystemExit(np.sum(np.array([1, 2, 3])))\n",
                target=target,
            )
            self.assertIn("not in the native subset", message)

    def test_import_torch_is_rejected(self):
        message = self._reject(
            "import torch\n"
            "raise SystemExit(torch.sum(torch.tensor([1, 2])))\n"
        )
        self.assertIn("not in the native subset", message)

    def test_from_numpy_import_is_rejected(self):
        message = self._reject(
            "from numpy import array, dot\n"
            "raise SystemExit(dot(array([1, 2]), array([3, 4])))\n"
        )
        self.assertIn("not in the native subset", message)

    def test_from_torch_functional_import_is_rejected(self):
        message = self._reject(
            "from torch.nn.functional import relu\n"
            "raise SystemExit(0)\n"
        )
        self.assertIn("not in the native subset", message)

    def test_rejection_message_explains_the_semantic_gap(self):
        message = self._reject(
            "import numpy as np\n"
            "raise SystemExit(np.sum(np.array([1, 2, 3])))\n"
        )
        # The message must explain *why*, not just refuse.
        self.assertIn("does not match", message)

    def test_experimental_kernels_flag_is_inert_via_api(self):
        # Passing the retired flag through the API does not resurrect the
        # substitution: numpy/torch remain rejected.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "prog.py"
            entry.write_text(
                "import numpy as np\n"
                "raise SystemExit(np.sum(np.array([1, 2, 3])))\n",
                encoding="utf-8",
            )
            with self.assertRaises(NativeCompileError):
                compile_native(
                    entry,
                    root / "prog",
                    "linux-x86_64",
                    experimental_kernels=True,
                )

    def test_library_audit_never_marks_numpy_helper_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "vectorlib"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "score.py").write_text(
                "import numpy as np\n"
                "def score(value: int) -> int:\n"
                "    return np.sum(np.array([value, -1]))\n",
                encoding="utf-8",
            )
            report = audit_native_library(package, source_roots=(root,))
            self.assertFalse(report.fully_native)
            # The retired flag must not change the verdict.
            report_with_flag = audit_native_library(
                package,
                source_roots=(root,),
                experimental_kernels=True,
            )
            self.assertFalse(report_with_flag.fully_native)

    def test_cli_compile_rejects_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "app.py"
            source.write_text(
                "import torch\n"
                "raise SystemExit(torch.tensor([-2, 6]).relu().sum())\n",
                encoding="utf-8",
            )
            output = root / "app.exe"
            status = main(
                [
                    "compile",
                    str(source),
                    "--target",
                    "windows-x86_64",
                    "--output",
                    str(output),
                ]
            )
            self.assertNotEqual(status, 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
