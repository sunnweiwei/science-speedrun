"""Candidate site/date selection must not depend on heldout target values."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from openpyxl import Workbook


class TestMetadata(unittest.TestCase):
    def test_targets_cannot_change_selection(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "docs/downstream/external_lfmc_metadata.py"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coords = root / "coords.npy"
            np.save(coords, np.array([[0.0, 0.0]]))
            policy = {
                "file_md5": "",
                "woody_functional_types": ["tree"],
                "coordinate_bounds": {
                    "min_lon": -20,
                    "max_lon": 180,
                    "min_lat": -60,
                    "max_lat": 75,
                },
                "eligible_dates": ["2018-01-01", "2023-12-31"],
                "minimum_distinct_dates_per_site": 3,
                "maximum_sites": 200,
                "maximum_dates_per_site": 12,
            }
            outputs = []
            for variant in range(2):
                book = Workbook()
                sheet = book.active
                sheet.title = "LFMC data"
                sheet.append(
                    [
                        "Latitude (WGS84, EPSG:4326)",
                        "Longitude (WGS84, EPSG:4326)",
                        "Sampling date (YYYYMMDD)",
                        "Species functional type",
                        "Country",
                        "LFMC value (%)",
                    ]
                )
                for day in [20180101, 20180201, 20180301]:
                    sheet.append(
                        [
                            -30,
                            150,
                            day,
                            "tree",
                            "Australia",
                            50 if variant == 0 else -99999,
                        ]
                    )
                    sheet.append(
                        [
                            0,
                            0,
                            day,
                            "tree",
                            "Excluded location",
                            50 if variant == 0 else 99999,
                        ]
                    )
                path = root / f"data{variant}.xlsx"
                book.save(path)
                with path.open("rb") as f:
                    policy["file_md5"] = hashlib.file_digest(f, "md5").hexdigest()
                policy_path = root / "policy.json"
                policy_path.write_text(json.dumps(policy))
                output = root / f"output{variant}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "--workbook",
                        str(path),
                        "--policy",
                        str(policy_path),
                        "--pretrain-coordinates",
                        str(coords),
                        "--output",
                        str(output),
                    ],
                    env=os.environ,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(json.loads(output.read_text()))
            self.assertEqual(outputs[0]["sites"], outputs[1]["sites"])
            self.assertEqual(outputs[0]["selected_sites"], 1)
            self.assertEqual(outputs[0]["selected_site_dates"], 3)
            self.assertFalse(outputs[0]["target_column_selected"])
            self.assertEqual(
                outputs[0]["source_counters"]["pretraining_buffer_excluded_rows"], 3
            )


if __name__ == "__main__":
    unittest.main()
