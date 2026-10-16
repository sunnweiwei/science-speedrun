"""A failed cohort cannot unlock test reporting; test labels cannot choose alpha."""

import hashlib
import json
import os
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class HeldoutConfirmationTest(unittest.TestCase):
    def test_legacy_random_provenance(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/analysis"))
        check = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "docs/analysis/heldout_confirmation.py")
        )["is_random_initialization"]
        self.assertTrue(check({"checkpoint": None}))
        self.assertTrue(check({"origin": "random_initialization"}))
        self.assertFalse(check({}))
        self.assertFalse(check({"checkpoint": "weights.pt"}))
        self.assertFalse(
            check(
                {"origin": "official_public_pretrained_reference", "checkpoint": None}
            )
        )

    def test_lock_gate_and_fixed_probe_parameters(self):
        script = Path(__file__).resolve().parents[1] / "docs/analysis/heldout_confirmation.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            context = {"gpu_type": "a5000", "python_hash_seed": 0}
            (root / "execution-context.json").write_text(json.dumps(context))
            seeds, loader_seeds = [12536, 12539, 12540], [3622, 3625, 3626]
            pairs = [
                [f"large-random-eval-{seed}", f"train-{seed}-step6000"]
                for seed in seeds
            ]
            recipe = {"global_batch": 64, "checkpoint_step": "step6000"}
            lock = {
                "selected": pairs[0][1],
                "checkpoint_step": 6000,
                "training_recipe": recipe,
                "policy": {"development_seed": seeds[0], "execution_context": context},
                "confirmation": [
                    {
                        "name": f"train-{seed}",
                        "initialization_seed": seed,
                        "data_seed": loader,
                        "paired_random": pair[0],
                    }
                    for seed, loader, pair in zip(
                        seeds[1:], loader_seeds[1:], pairs[1:]
                    )
                ],
            }
            comparison = {
                "split": "val",
                "pairs": pairs,
                "initialization_seeds": seeds,
                "data_seeds": loader_seeds,
                "execution_context": context,
                "tasks": {"lfmc": {"three_seed_criterion_met": False}},
            }
            lock_path, comparison_path = root / "lock.json", root / "comparison.json"
            lock_path.write_text(json.dumps(lock))
            comparison_path.write_text(json.dumps(comparison))
            cmd = [
                sys.executable,
                str(script),
                "--data-root",
                str(root),
                "--lock",
                str(lock_path),
                "--validation-comparison",
                str(comparison_path),
                "--output",
                str(root / "test.json"),
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            failed = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("Validation confirmation has not passed", failed.stderr)
            self.assertFalse((root / "eval").exists())
            rng = np.random.default_rng(12)
            y = rng.normal(size=100) * 3
            ids = np.array([f"id-{i}" for i in range(len(y))])
            rows = [
                {
                    "id": str(id_),
                    "label": float(y[i]),
                    "split": "train" if i < 60 else "val" if i < 80 else "test",
                    "latlon": [float(i), 0.0],
                }
                for i, id_ in enumerate(ids)
            ]
            labels = root / "eval/lfmc/labels.json"
            labels.parent.mkdir(parents=True)
            labels.write_text(json.dumps(rows))
            names = [n for pair in pairs for n in pair] + ["large-raw"]
            for name in names:
                x = rng.normal(size=(len(y), 8))
                if name.startswith("train-") or name == "large-raw":
                    x[:, 0] = y
                feature = root / "features" / f"{name}-lfmc" / "features.npz"
                feature.parent.mkdir(parents=True)
                np.savez(feature, ids=ids, features=x.astype(np.float32))
                x = x.astype(np.float32).astype(np.float64)
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
                    x[:60], y[:60]
                )
                report = {
                    "features": str(feature),
                    "probe": {
                        "selected_parameter": 1.0,
                        "validation": {
                            "mae": float(
                                np.abs(model.predict(x[60:80]) - y[60:80]).mean()
                            )
                        },
                    },
                }
                (root / "results" / f"{name}-lfmc.json").write_text(json.dumps(report))
            for seed, loader, pair in zip(seeds, loader_seeds, pairs):
                for index, name in enumerate(pair):
                    package = root / "packages" / name
                    package.mkdir(parents=True)
                    (package / "encoder.pt").write_bytes(name.encode())
                    sha = hashlib.sha256(name.encode()).hexdigest()
                    provenance = {
                        "origin": "random_initialization"
                        if index == 0
                        else "own_from_scratch_training",
                        "initialization_seed": seed,
                        "data_seed": loader if index else None,
                        "training_recipe": recipe if index else None,
                        "variant": "v1/large",
                        "parameters": 307770368,
                        "infer_sha256": "fixture",
                        "encoder_sha256": sha,
                    }
                    if index == 0 and seed == seeds[0]:
                        del provenance["origin"]
                        provenance["checkpoint"] = None
                    (package / "provenance.json").write_text(json.dumps(provenance))
                    if name == lock["selected"]:
                        lock["encoder_sha256"] = sha
            lock_path.write_text(json.dumps(lock))
            comparison["tasks"]["lfmc"]["three_seed_criterion_met"] = True
            comparison["tasks"]["lfmc"].update(
                per_seed_gain=[1.0, 1.0, 1.0],
                mean_seed_gain=1.0,
                spatial_bootstrap_95_percent_interval=[0.1, 2.0],
                seed_and_spatial_bootstrap_95_percent_interval=[0.1, 2.0],
            )
            comparison["tasks"]["lfmc"][
                "seed_and_spatial_bootstrap_95_percent_interval"
            ][0] = -0.1
            comparison_path.write_text(json.dumps(comparison))
            failed = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("flag disagrees", failed.stderr)
            self.assertFalse((root / "test.json").exists())
            comparison["tasks"]["lfmc"][
                "seed_and_spatial_bootstrap_95_percent_interval"
            ][0] = 0.1
            comparison_path.write_text(json.dumps(comparison))
            run = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            first = json.loads((root / "test.json").read_text())
            self.assertEqual(first["paired_random_comparison"]["rows"], 20)
            self.assertEqual(first["paired_random_comparison"]["groups"], 20)
            repeated = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            for row in rows[80:]:
                row["label"] += 100
            labels.write_text(json.dumps(rows))
            cmd[-1] = str(root / "changed-test-labels.json")
            run = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            altered = json.loads((root / "changed-test-labels.json").read_text())
            for name in names:
                self.assertEqual(
                    first["models"][name]["selected_alpha_from_validation"],
                    altered["models"][name]["selected_alpha_from_validation"],
                )
                self.assertEqual(
                    first["models"][name]["validation_mae_check"],
                    altered["models"][name]["validation_mae_check"],
                )
                self.assertNotEqual(
                    first["models"][name]["test_mae"],
                    altered["models"][name]["test_mae"],
                )
            changed = root / "packages" / pairs[1][1] / "provenance.json"
            provenance = json.loads(changed.read_text())
            provenance["training_recipe"] = {"different": True}
            changed.write_text(json.dumps(provenance))
            cmd[-1] = str(root / "bad-recipe.json")
            run = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("Training recipe differs", run.stderr)
            self.assertFalse((root / "bad-recipe.json").exists())


if __name__ == "__main__":
    unittest.main()
