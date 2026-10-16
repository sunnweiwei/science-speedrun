"""Never extract heldout payloads when preparing downstream development data."""

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from extract_trainval import extract


class TestExtract(unittest.TestCase):
    def test_heldout_payload_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "source.tar"
            with tarfile.open(archive, "w") as tar:
                files = {"lfmc/config.json": b"{}"}
                for split in ["train", "val", "test"]:
                    prefix = f"lfmc/windows/spatial_split/{split}"
                    files[prefix + "/metadata.json"] = json.dumps(
                        {"options": {"split": split}}
                    ).encode()
                    files[prefix + "/layers/labels/value/geotiff.tif"] = b"payload"
                # Real release may place payloads before metadata.
                for name, data in reversed(list(files.items())):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            original = tarfile.TarFile.extractfile
            accesses = []

            def track(tar, member):
                accesses.append(member.name)
                return original(tar, member)

            with patch.object(tarfile.TarFile, "extractfile", track):
                report = extract(
                    archive, root / "output", {"train": 1, "val": 1, "test": 1}
                )
            self.assertFalse((root / "output/windows/spatial_split/test").exists())
            self.assertTrue(
                (
                    root
                    / "output/windows/spatial_split/train/layers/labels/value/geotiff.tif"
                ).exists()
            )
            self.assertFalse(any("/test/layers/" in n for n in accesses))
            self.assertFalse(report["heldout_window_payloads_read"])
            with self.assertRaises(ValueError):
                extract(archive, root / "output")


if __name__ == "__main__":
    unittest.main()
