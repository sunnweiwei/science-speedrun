"""Check declared task ranking, context isolation, and immutable selection records."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SelectionTest(unittest.TestCase):
    def test_task_policy_and_guards(self):
        script = Path(__file__).resolve().parents[1] / "docs/analysis/select_checkpoint.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            context = {"gpu_type": "a5000", "python_hash_seed": 0}
            (root / "execution-context.json").write_text(json.dumps(context))
            names = ["better-awf-step500", "better-lfmc-step1000"]
            for name, acc, mae in zip(names, [0.9, 0.5], [23, 21]):
                package = root / "packages" / name
                package.mkdir(parents=True)
                (package / "encoder.pt").write_bytes(name.encode())
                (package / "provenance.json").write_text(
                    json.dumps(
                        {
                            "origin": "own_from_scratch_training",
                            "checkpoint": "own-checkpoint",
                            "encoder_sha256": hashlib.sha256(name.encode()).hexdigest(),
                        }
                    )
                )
                for task in ["awf", "lfmc"]:
                    report = {
                        "features": str(
                            root / "features" / f"{name}-{task}" / "features.npz"
                        ),
                        "validation_predictions": str(
                            root / "results" / f"{name}-{task}-predictions.npz"
                        ),
                        "probe": {
                            "validation": {"accuracy": acc}
                            if task == "awf"
                            else {"mae": mae}
                        },
                        "constant_val": {"mae": 30},
                    }
                    (root / "results" / f"{name}-{task}.json").write_text(
                        json.dumps(report)
                    )
            policy = {"candidates": names, "execution_context": context}
            policy_path = root / "policy.json"
            output = root / "selection.json"
            cmd = [
                sys.executable,
                str(script),
                "--data-root",
                str(root),
                "--policy",
                str(policy_path),
                "--output",
                str(output),
            ]
            for primary, expected in [("joint", names[0]), ("lfmc", names[1])]:
                policy["primary_task"] = primary
                policy_path.write_text(json.dumps(policy))
                run = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                selected = json.loads(output.read_text())
                self.assertEqual(selected["selected"], expected)
                self.assertTrue(selected["ranking_only"])
                repeated = subprocess.run(cmd, capture_output=True, text=True)
                self.assertNotEqual(repeated.returncode, 0)
                self.assertEqual(json.loads(output.read_text()), selected)
                output.unlink()
            metric_path = root / "results" / f"{names[1]}-lfmc.json"
            metric = json.loads(metric_path.read_text())
            metric["features"] = "/different-context/features.npz"
            metric_path.write_text(json.dumps(metric))
            rejected = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("execution contexts", rejected.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
