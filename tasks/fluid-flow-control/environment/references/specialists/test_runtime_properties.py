import hashlib
from pathlib import Path
import tempfile
import tomllib
import unittest

from native import runtime_properties


class CapacityTests(unittest.TestCase):
    def test_private_capacity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "published.toml"
            payload = b"Re = 150\nmaxNoCells = 80000000 # per process\nmaxRfnmntLvl = 12\ngridGenerator = false\n"
            source.write_bytes(payload)
            output = root / "output"
            output.mkdir()
            destination, audit = runtime_properties(source, output, 20000000)
            original = tomllib.loads(payload.decode())
            changed = tomllib.loads(destination.read_text())
            original["maxNoCells"] = 20000000
            self.assertEqual(original, changed)
            self.assertEqual(source.read_bytes(), payload)
            self.assertEqual(audit["source_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(audit["runtime_sha256"], hashlib.sha256(destination.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                runtime_properties(source, output, 30000000)

    def test_default_does_not_copy(self):
        path = Path("/nonexistent/published.toml")
        self.assertEqual(runtime_properties(path, path.parent, None), (path, None))

    def test_rejects_ambiguous_property(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "published.toml"
            for payload in ("Re = 150\n", "maxNoCells=80\nmaxNoCells=90\n"):
                source.write_text(payload)
                with self.assertRaises(ValueError):
                    runtime_properties(source, root, 20)
                self.assertFalse((root / "runtime-properties.toml").exists())

    def test_rejects_invalid_capacity(self):
        for capacity in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                runtime_properties("/not-read", "/not-written", capacity)


if __name__ == "__main__":
    unittest.main()
