import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from hf_snapshot import DEFAULT_ENDPOINT, endpoint, verify_tree


class VerifyTreeTest(unittest.TestCase):
    def test_default_endpoint_is_mirror(self):
        self.assertEqual(DEFAULT_ENDPOINT, "https://hf-mirror.com")

    def test_endpoint_environment_override(self):
        import os

        old_value = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = "https://example.invalid/"
            self.assertEqual(endpoint(), "https://example.invalid")
        finally:
            if old_value is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = old_value

    def test_lfs_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "model.safetensors"
            path.write_bytes(b"sample weights")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            entry = SimpleNamespace(
                path=path.name,
                size=path.stat().st_size,
                lfs={"sha256": sha},
            )
            report = verify_tree(directory, [entry])
            self.assertEqual(report[0]["sha256"], sha)
            entry.lfs = {"sha256": "0" * 64}
            with self.assertRaises(ValueError):
                verify_tree(directory, [entry])


if __name__ == "__main__":
    unittest.main()
