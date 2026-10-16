"""Checks ID alignment, rejection of malformed output, and test isolation."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "environment/evaluation/probes.py"


class ProbeContract(unittest.TestCase):
    def test_alignment_failures_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(24)
            x = rng.normal(size=(110, 12))
            y = x[:, 0] * 4 + x[:, 1] * 2
            rows = [
                {
                    "id": f"new-id-{i}",
                    "label": float(y[i]),
                    "split": "train" if i < 70 else "val" if i < 90 else "test",
                }
                for i in range(110)
            ]
            labels = root / "labels.json"
            labels.write_text(json.dumps(rows))
            ids = np.array([r["id"] for r in rows])
            features = root / "features.npz"
            cmd = [
                sys.executable,
                str(SCRIPT),
                "--features",
                str(features),
                "--labels",
                str(labels),
                "--task",
                "lfmc",
                "--output",
                str(root / "score.json"),
                "--final-test",
            ]

            def run():
                return subprocess.run(cmd, capture_output=True, text=True)

            np.savez(features, ids=ids, features=x)
            self.assertEqual(run().returncode, 0)
            original = json.loads((root / "score.json").read_text())
            perm = rng.permutation(len(ids))
            np.savez(features, ids=ids[perm], features=x[perm])
            self.assertEqual(run().returncode, 0)
            self.assertEqual(original, json.loads((root / "score.json").read_text()))
            for row in rows[90:]:
                row["label"] += 100
            labels.write_text(json.dumps(rows))
            self.assertEqual(run().returncode, 0)
            altered = json.loads((root / "score.json").read_text())
            self.assertEqual(
                original["probe"]["selected_parameter"],
                altered["probe"]["selected_parameter"],
            )
            self.assertEqual(
                original["probe"]["validation"], altered["probe"]["validation"]
            )
            self.assertNotEqual(original["probe"]["test"], altered["probe"]["test"])
            np.savez(features, ids=ids[:-1], features=x[:-1])
            self.assertNotEqual(run().returncode, 0)
            np.savez(features, ids=np.repeat(ids[:1], len(ids)), features=x)
            self.assertNotEqual(run().returncode, 0)
            np.savez(
                features,
                ids=np.concatenate([ids, ids[:1]]),
                features=np.concatenate([x, x[:1]]),
            )
            self.assertNotEqual(run().returncode, 0)
            x[0, 0] = float("nan")
            np.savez(features, ids=ids, features=x)
            self.assertNotEqual(run().returncode, 0)


if __name__ == "__main__":
    unittest.main()
