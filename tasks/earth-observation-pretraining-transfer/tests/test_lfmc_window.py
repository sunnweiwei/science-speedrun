"""Original-window reader preserves off-center labels and sparse time/band groups."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/downstream"))
from lfmc_window import read_window


class TestWindow(unittest.TestCase):
    def test_sparse_groups_centering_and_heldout_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = {
                "name": "fixture",
                "bounds": [0, -63, 63, 0],
                "projection": {
                    "crs": "EPSG:32611",
                    "x_resolution": 10,
                    "y_resolution": -10,
                },
                "options": {"split": "train"},
            }
            (root / "metadata.json").write_text(json.dumps(meta))

            def write(relative, values):
                p = root / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(
                    p,
                    "w",
                    driver="GTiff",
                    height=63,
                    width=63,
                    count=len(values),
                    dtype=str(values.dtype),
                    crs="EPSG:32611",
                    transform=from_origin(0, 630, 10, 10),
                ) as d:
                    d.write(values)

            target = np.full((1, 63, 63), -1, dtype=np.float32)
            target[0, 30, 31] = 50
            write("layers/labels/value/geotiff.tif", target)
            write(
                "layers/sentinel2.3/B02_B03_B04_B08/geotiff.tif",
                np.full((4, 63, 63), 2000, dtype=np.uint16),
            )
            a = read_window(root, center_on_label=True)
            self.assertEqual(a["bounds"], (15, -49, 47, -17))
            self.assertEqual(a["target"][16, 16], 50)
            self.assertEqual(a["target_valid"].sum(), 1)
            self.assertEqual(a["valid"]["sentinel2_l2a"].sum(), 1)
            self.assertTrue(a["valid"]["sentinel2_l2a"][3, 0])
            self.assertFalse(a["valid"]["sentinel1"].any())
            self.assertEqual(a["pixels"]["sentinel2_l2a"].shape, (32, 32, 12, 12))
            empty = read_window(root, bounds=(0, -32, 32, 0))
            self.assertFalse(empty["target_valid"].any())
            meta["options"]["split"] = "test"
            (root / "metadata.json").write_text(json.dumps(meta))
            with patch(
                "lfmc_window.GeotiffRasterFormat.decode_raster",
                side_effect=AssertionError("Test payload read"),
            ):
                with self.assertRaisesRegex(ValueError, "heldout"):
                    read_window(root, center_on_label=True)


if __name__ == "__main__":
    unittest.main()
