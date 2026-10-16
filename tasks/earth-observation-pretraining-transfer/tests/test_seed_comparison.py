"""Statistical contract: paired seeds, stored scores, and consistent gains."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "environment/evaluation/compare_seeds.py"


class SeedComparisonTest(unittest.TestCase):
    def test_confirmation_and_pairing_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            pairs = []
            for seed in range(3):
                names = [f"random-{seed}", f"trained-{seed}"]
                pairs.append(":".join(names))
                for name in names:
                    package = root / "packages" / name
                    package.mkdir(parents=True)
                    (package / "provenance.json").write_text(
                        json.dumps(
                            {
                                "checkpoint": None
                                if name.startswith("random")
                                else "own-checkpoint",
                                "initialization_seed": seed,
                                "data_seed": seed + 100,
                                "variant": "test-fixture",
                                "parameters": 1,
                                "infer_sha256": "same-adapter",
                                "training_recipe": {"steps": 10},
                            }
                        )
                    )
            for task in ["awf", "lfmc"]:
                directory = root / "eval" / task
                directory.mkdir(parents=True)
                rows = [
                    {
                        "id": str(i),
                        "label": 0,
                        "split": "train" if i < 2 else "val",
                        "latlon": [float(i), float(i)],
                    }
                    for i in range(10)
                ]
                (directory / "labels.json").write_text(json.dumps(rows))
                for pair in pairs:
                    for name in pair.split(":"):
                        trained = name.startswith("trained")
                        prediction = np.full(8, 0 if trained else 1)
                        path = root / "results" / f"{name}-{task}-predictions.npz"
                        np.savez(
                            path,
                            ids=np.array([str(i) for i in range(2, 10)]),
                            predictions=prediction,
                        )
                        metric = (
                            {"accuracy": 1 if trained else 0}
                            if task == "awf"
                            else {"mae": 0 if trained else 1}
                        )
                        (root / "results" / f"{name}-{task}.json").write_text(
                            json.dumps(
                                {
                                    "probe": {"validation": metric},
                                    "validation_predictions": str(path),
                                }
                            )
                        )
            output = root / "comparison.json"

            def run(selected):
                return subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--data-root",
                        str(root),
                        "--pairs",
                        *selected,
                        "--raw-baseline",
                        "trained-0",
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                )

            success = run(pairs)
            self.assertEqual(success.returncode, 0, success.stderr)
            report = json.loads(output.read_text())
            for task in ["awf", "lfmc"]:
                self.assertTrue(report["tasks"][task]["three_seed_criterion_met"])
                raw = report["tasks"][task]["raw_baseline_comparison"]
                self.assertEqual(raw["mean_seed_gain"], 0)
                self.assertFalse(raw["three_seed_criterion_met"])
                self.assertEqual(
                    report["tasks"][task][
                        "seed_and_spatial_bootstrap_95_percent_interval"
                    ],
                    [1, 1],
                )
            (root / "execution-context.json").write_text('{"python_hash_seed": 0}')
            for pair in pairs:
                for name in pair.split(":"):
                    for task in ["awf", "lfmc"]:
                        score_path = root / "results" / f"{name}-{task}.json"
                        score = json.loads(score_path.read_text())
                        score["features"] = str(
                            root / "features" / f"{name}-{task}" / "features.npz"
                        )
                        score_path.write_text(json.dumps(score))
            same_context = run(pairs)
            self.assertEqual(same_context.returncode, 0, same_context.stderr)
            score_path = root / "results/trained-0-awf.json"
            original_score = score_path.read_text()
            score = json.loads(original_score)
            score["features"] = str(root / "other-context/features.npz")
            score_path.write_text(json.dumps(score))
            mixed = run(pairs)
            self.assertNotEqual(mixed.returncode, 0)
            self.assertIn("Cannot mix execution contexts", mixed.stderr)
            score_path.write_text(original_score)
            duplicate = run([pairs[0], pairs[0], pairs[0]])
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn("Repeated initialization seeds", duplicate.stderr)
            path = root / "results/trained-2-awf-predictions.npz"
            np.savez(
                path,
                ids=np.array([str(i) for i in range(2, 10)]),
                predictions=np.ones(8),
            )
            mismatch = run(pairs)
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertIn("Saved predictions disagree", mismatch.stderr)
            provenance = root / "packages/trained-0/provenance.json"
            invalid = json.loads(provenance.read_text())
            invalid["training_validity"] = "invalid"
            provenance.write_text(json.dumps(invalid))
            rejected = run(pairs)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Known invalid training implementation", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
