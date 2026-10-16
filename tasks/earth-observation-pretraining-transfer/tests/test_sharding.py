"""Check lossless input shards, complete batch boundaries and merge rejection."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

TRACK = Path(__file__).resolve().parents[1]


class ShardingTest(unittest.TestCase):
    def test_coverage_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.h5"
            pixels = np.arange(102, dtype=np.uint16).reshape(51, 2)
            with h5py.File(source, "w") as h:
                h.attrs["task"] = "lfmc"
                h.create_dataset(
                    "ids", data=[str(i) for i in range(51)], dtype=h5py.string_dtype()
                )
                h.create_dataset("sentinel2_l2a", data=pixels)
            command = [
                sys.executable,
                str(TRACK / "environment/data/shard_inputs.py"),
                "--input",
                str(source),
                "--output",
                str(root / "shards"),
                "--shards",
                "3",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_path = root / "shards/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual([s["start"] for s in manifest["shards"]], [0, 16, 32])
            collected = []
            for index, shard in enumerate(manifest["shards"]):
                with h5py.File(root / "shards" / shard["file"]) as h:
                    collected.append(h["sentinel2_l2a"][:])
                    ids = h["ids"].asstr()[:].astype(str)
                    features = np.repeat(
                        np.arange(shard["start"], shard["stop"], dtype=np.float32)[
                            :, None
                        ],
                        3,
                        axis=1,
                    )
                part = root / f"features/part{index:03d}"
                part.mkdir(parents=True)
                np.savez(part / "features.npz", ids=ids, features=features)
            np.testing.assert_array_equal(np.concatenate(collected), pixels)
            output = root / "merged.npz"
            merge = [
                sys.executable,
                str(TRACK / "docs/analysis/merge_features.py"),
                "--manifest",
                str(manifest_path),
                "--features",
                str(root / "features"),
                "--output",
                str(output),
            ]
            result = subprocess.run(merge, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(output, allow_pickle=False) as merged:
                np.testing.assert_array_equal(merged["ids"], np.arange(51).astype(str))
                np.testing.assert_array_equal(merged["features"][:, 0], np.arange(51))
            result = subprocess.run(merge, capture_output=True, text=True)
            self.assertNotEqual(
                result.returncode, 0, "must not overwrite existing features"
            )
            output.unlink()
            original = manifest_path.read_text()
            manifest["start"] = 16
            manifest["full_release"] = False
            manifest_path.write_text(json.dumps(manifest))
            result = subprocess.run(merge, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("complete release", result.stderr)
            manifest_path.write_text(original)
            path = root / "features/part001/features.npz"
            with np.load(path, allow_pickle=False) as good:
                ids, features = good["ids"].copy(), good["features"].copy()
            ids[0] = ids[1]
            np.savez(path, ids=ids, features=features)
            result = subprocess.run(merge, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("IDs do not match", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
