"""Deferred scoring must use the exact label-free independent replay."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import grade


class StagedVerifierTest(unittest.TestCase):
    def test_label_free_replay_tamper_rejection_and_real_cpu_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, data, output = (root / n for n in ["model", "data", "output"])
            model.mkdir()
            for name in ["encoder.pt", "config.json", "infer.py", "LICENSE-OlmoEarth"]:
                (model / name).write_text("fixture")
            (model / "provenance.json").write_text(
                json.dumps({"origin": "own_from_scratch_training"})
            )
            (model / "config.json").write_text("{}")
            probes = root / "probes.py"
            shutil.copyfile(
                Path(__file__).resolve().parents[1]
                / "environment/evaluation/probes.py",
                probes,
            )
            for task in ["awf", "lfmc"]:
                (data / task).mkdir(parents=True)
                (data / task / "inputs.h5").write_bytes(b"unlabeled-input-fixture")
            rng = np.random.default_rng(17)
            x = rng.normal(size=(110, 8)).astype(np.float32)
            ids = np.array([f"id-{i}" for i in range(len(x))])
            command = [
                "grade.py",
                "--model",
                str(model),
                "--data",
                str(data),
                "--output",
                str(output),
                "--probes",
                str(probes),
            ]

            def fake_replay(cmd, **kwargs):
                self.assertTrue(
                    cmd[1].endswith("replay.py"), "Inference-only attempted scoring"
                )
                target = Path(cmd[cmd.index("--output") + 1])
                np.savez(target / "features.npz", ids=ids, features=x)
                return subprocess.CompletedProcess(cmd, 0)

            with (
                patch.object(sys, "argv", command + ["--inference-only"]),
                patch.object(grade.subprocess, "run", side_effect=fake_replay),
            ):
                grade.main()
            manifest = json.loads((output / "inference-manifest.json").read_text())
            self.assertFalse(manifest["labels_read"])
            self.assertFalse((output / "reward.json").exists())
            self.assertFalse(any(data.glob("*/labels.json")))
            for path in [
                model / "encoder.pt",
                model / "infer.py",
                data / "lfmc/inputs.h5",
                output / "lfmc/features.npz",
                probes,
            ]:
                original = path.read_bytes()
                path.write_bytes(original + b"tampered")
                with self.assertRaisesRegex(
                    ValueError, "artifacts or verifier code changed"
                ):
                    grade.validate_manifest(manifest, model, data, output, probes)
                path.write_bytes(original)
            for task in ["awf", "lfmc"]:
                rows = [
                    {
                        "id": id_,
                        "label": int(x[i, 0] > 0)
                        if task == "awf"
                        else float(4 * x[i, 0] + 2 * x[i, 1]),
                        "split": "train"
                        if i < 70
                        else "val"
                        if task == "awf" or i < 90
                        else "test",
                    }
                    for i, id_ in enumerate(ids)
                ]
                (data / task / "labels.json").write_text(json.dumps(rows))
            real_run = subprocess.run

            def score_only(cmd, **kwargs):
                self.assertEqual(
                    Path(cmd[1]), probes, "Deferred scoring attempted inference"
                )
                return real_run(cmd, **kwargs, capture_output=True, text=True)

            with (
                patch.object(sys, "argv", command + ["--score-existing"]),
                patch.object(grade.subprocess, "run", side_effect=score_only),
                patch.dict(
                    os.environ,
                    {
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                ),
            ):
                grade.main()
            result = json.loads((output / "reward.json").read_text())
            self.assertTrue(all(isinstance(v, (int, float)) for v in result.values()))
            self.assertEqual(
                json.loads((output / "report.json").read_text())["inference_manifest_sha256"],
                grade.digest(output / "inference-manifest.json"),
            )
            self.assertEqual(
                result["reward"],
                0.5
                * (result["awf_validation_accuracy"] + result["lfmc_test_mae_skill"]),
            )
            with patch.object(sys, "argv", command + ["--score-existing"]):
                with self.assertRaisesRegex(ValueError, "Scoring already completed"):
                    grade.main()
            full_output = root / "full-output"
            full_command = command.copy()
            full_command[full_command.index("--output") + 1] = str(full_output)

            def full_run(cmd, **kwargs):
                if cmd[1].endswith("replay.py"):
                    return fake_replay(cmd, **kwargs)
                return score_only(cmd, **kwargs)

            with (
                patch.object(sys, "argv", full_command),
                patch.object(grade.subprocess, "run", side_effect=full_run),
                patch.dict(
                    os.environ,
                    {
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                ),
            ):
                grade.main()
            full_result = json.loads((full_output / "reward.json").read_text())
            for key in [
                "reward",
                "awf_validation_accuracy",
                "lfmc_test_mae",
                "lfmc_test_mae_skill",
            ]:
                self.assertAlmostEqual(result[key], full_result[key], places=12)
            for provenance, message in [
                ({"training_validity": "invalid"}, "invalid training"),
                ({"origin": "official_public_pretrained_reference"}, "ineligible"),
            ]:
                (model / "provenance.json").write_text(json.dumps(provenance))
                invalid_command = command.copy()
                invalid_output = root / ("invalid-" + str(len(message)))
                invalid_command[invalid_command.index("--output") + 1] = str(invalid_output)
                with patch.object(sys, "argv", invalid_command + ["--inference-only"]):
                    grade.main()
                invalid = json.loads((invalid_output / "reward.json").read_text())
                self.assertEqual(invalid["reward"], 0)
                self.assertIn(message, json.loads((invalid_output / "report.json").read_text())["reason"])


if __name__ == "__main__":
    unittest.main()
