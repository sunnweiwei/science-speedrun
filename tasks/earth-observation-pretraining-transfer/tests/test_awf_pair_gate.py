"""The full-run gate rejects incomplete or mismatched integration pilots."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/analysis"))
from verify_awf_pair import verify


class TestGate(unittest.TestCase):
    def test_complete_pair_and_failed_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = {
                "stage": "integration pilot, not scientific evidence",
                "outputs": ["trained", "random"],
                "package_names": {
                    "trained": "trained-package",
                    "random": "random-package",
                },
                "head_and_loader_seed": 8123,
                "validation_rng_seed": 73491,
                "source_sha256": {
                    "environment/downstream/train_awf.py": "train",
                    "environment/downstream/portable_encoder.py": "adapter",
                    "results/protocol-audit/awf-large-candidate.yaml": "config",
                },
            }
            for name in ["trained", "random"]:
                out = root / name
                out.mkdir()
                package = root / f"{name}-package"
                package.mkdir()
                provenance = {
                    "origin": "own_from_scratch_training"
                    if name == "trained"
                    else "random_initialization",
                    "encoder_sha256": name,
                    "initialization_seed": 12536,
                    "variant": "v1/large",
                    "parameters": 307770368,
                }
                (package / "provenance.json").write_text(json.dumps(provenance))
                values = {
                    "settings": {
                        "pilot": True,
                        "max_epochs": 22,
                        "seed": 8123,
                        "limit_train_batches": 2,
                        "limit_val_batches": 2,
                        "num_workers": 0,
                        "validation_rng_seed": 73491,
                        "test_used": False,
                        "train_source_sha256": "train",
                        "adapter_source_sha256": "adapter",
                        "config_sha256": "config",
                        "package": str(package),
                        "dataset": "same-dataset",
                    },
                    "initialization": {
                        "encoder_file_sha256": name,
                        "encoder_state_digest": name,
                        "head_state_digest": "same-head",
                    },
                    "data-coverage": {"train": 1115, "val": 344},
                    "unfreeze-boundary": {
                        "epoch": 20,
                        "frozen_weights_unchanged": True,
                    },
                    "completion": {
                        "epochs": 22,
                        "encoder_changed": True,
                        "final_encoder_state_digest": name + "-changed",
                        "test_used": False,
                    },
                    "epochs": [
                        {
                            "epoch": i,
                            "encoder_trainable_parameters": 0 if i < 20 else 307770368,
                            "lr": [1e-4] if i < 20 else [1e-4, 1e-5],
                        }
                        for i in range(22)
                    ],
                }
                for k, v in values.items():
                    (out / f"{k}.json").write_text(json.dumps(v))
            self.assertTrue(verify(root, receipt)["gate_passed"])
            cases = [
                ("completion", "encoder_changed", False),
                ("initialization", "head_state_digest", "different-head"),
                ("unfreeze-boundary", "frozen_weights_unchanged", False),
                ("settings", "test_used", True),
                ("settings", "adapter_source_sha256", "changed-code"),
            ]
            for file, key, value in cases:
                p = root / "random" / f"{file}.json"
                original = p.read_text()
                changed = json.loads(original)
                changed[key] = value
                p.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    verify(root, receipt)
                p.write_text(original)
            p = root / "random/epochs.json"
            epochs = json.loads(p.read_text())
            p.write_text(json.dumps(epochs[:20]))
            with self.assertRaisesRegex(ValueError, "full unfreeze"):
                verify(root, receipt)


if __name__ == "__main__":
    unittest.main()
