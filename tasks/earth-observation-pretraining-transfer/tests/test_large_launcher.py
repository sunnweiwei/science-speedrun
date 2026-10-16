"""Verify job dependencies, explicit Large settings and duplicate submission guard."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LargeLauncherTest(unittest.TestCase):
    def test_submission_graph_without_slurm(self):
        launcher = Path(__file__).resolve().parents[1] / "docs/slurm/launch_large.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data with spaces"
            (data / "9923").mkdir(parents=True)
            runtime = root / "upstream"
            (runtime / ".venv/bin").mkdir(parents=True)
            (runtime / ".venv/bin/python").touch()
            fake = root / "sbatch"
            fake.write_text(
                f"#!{sys.executable}\n"
                "import json,os,sys\n"
                "from pathlib import Path\n"
                "p=Path(os.environ['FAKE_SLURM_LOG'])\n"
                "calls=json.loads(p.read_text()) if p.exists() else []\n"
                "calls.append({'argv':sys.argv[1:], 'settings':{k:v for k,v in os.environ.items() if k.startswith('OLMO_')}})\n"
                "p.write_text(json.dumps(calls))\n"
                "print(1000+len(calls))\n"
            )
            fake.chmod(0o755)
            log = root / "submitted.json"
            env = dict(
                os.environ, PATH=f"{root}:{os.environ['PATH']}", FAKE_SLURM_LOG=str(log)
            )
            command = [
                sys.executable,
                str(launcher),
                "--data",
                str(data),
                "--upstream",
                str(runtime),
                "--name",
                "confirmation",
                "--seed",
                "12537",
                "--data-seed",
                "3623",
            ]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = json.loads(log.read_text())
            self.assertEqual(len(calls), 2)
            self.assertIn("--dependency=afterok:1001", calls[1]["argv"])
            for call in calls:
                self.assertIn("--partition=" + os.environ.get("EARTH_SLURM_PARTITION", "gpu"), call["argv"])
                self.assertIn("--gres=gpu:rtx_pro_6000:1", call["argv"])
                self.assertEqual(call["settings"]["OLMO_VARIANT"], "large")
                self.assertEqual(call["settings"]["OLMO_SEED"], "12537")
                self.assertEqual(call["settings"]["OLMO_DATA_SEED"], "3623")
                self.assertEqual(call["settings"]["OLMO_DATA"], str(data))
            receipt = json.loads((data / "launches/confirmation.json").read_text())
            self.assertEqual(receipt["validation_job"], 1002)
            repeated = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(len(json.loads(log.read_text())), 2)
            two_gpu = command.copy()
            two_gpu[two_gpu.index("--name") + 1] = "two-gpu"
            two_gpu += ["--gpus", "2"]
            result = subprocess.run(two_gpu, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = json.loads(log.read_text())
            self.assertIn("--gres=gpu:rtx_pro_6000:2", calls[2]["argv"])
            self.assertIn("--gres=gpu:rtx_pro_6000:1", calls[3]["argv"])
            self.assertIn("--dependency=afterok:1003", calls[3]["argv"])
            receipt = json.loads((data / "launches/two-gpu.json").read_text())
            self.assertEqual(receipt["settings"]["NCCL_P2P_DISABLE"], "1")
            self.assertEqual(receipt["settings"]["OLMO_BATCH"], "64")
            self.assertEqual(receipt["settings"]["OLMO_MICROBATCH"], "32")
            early = two_gpu.copy()
            early[early.index("--name") + 1] = "early-checkpoint"
            early += ["--evaluate-step", "500"]
            result = subprocess.run(early, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((data / "launches/early-checkpoint.json").read_text())
            settings = receipt["settings"]
            self.assertEqual(settings["OLMO_STEPS"], "2000")
            self.assertEqual(settings["OLMO_NAME"], "early-checkpoint-step500")
            self.assertEqual(settings["OLMO_FIXED_CHECKPOINT_STEP"], "500")
            self.assertGreaterEqual(int(settings["OLMO_KEEP_CHECKPOINTS"]), 4)
            self.assertTrue(
                settings["OLMO_CHECKPOINT"].endswith("/step500/model_and_optim")
            )
            early[-1] = "375"
            result = subprocess.run(early, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(json.loads(log.read_text())), 6)
            (data / "19837").mkdir()
            expanded = two_gpu.copy()
            expanded[expanded.index("--name") + 1] = "two-shards"
            expanded += ["--pretrain-source", "two-shards"]
            result = subprocess.run(expanded, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((data / "launches/two-shards.json").read_text())
            self.assertEqual(
                receipt["settings"]["OLMO_PRETRAIN_DATA"], str(data / "19837")
            )
            native = two_gpu.copy()
            native[native.index("--name") + 1] = "native-batch"
            native += [
                "--global-batch",
                "512",
                "--steps",
                "750",
                "--warmup",
                "38",
                "--lr",
                "0.0001",
            ]
            result = subprocess.run(native, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((data / "launches/native-batch.json").read_text())
            settings = receipt["settings"]
            self.assertEqual(settings["OLMO_BATCH"], "512")
            self.assertEqual(settings["OLMO_STEPS"], "750")
            self.assertEqual(settings["OLMO_EPHEMERAL_INTERVAL"], "250")
            evaluation = root / "a5000 evaluation"
            (evaluation / "eval").mkdir(parents=True)
            (evaluation / "packages").mkdir()
            alternate = two_gpu.copy()
            alternate[alternate.index("--name") + 1] = "alternate-evaluator"
            alternate += [
                "--evaluation-gpu",
                "a5000",
                "--evaluation-root",
                str(evaluation),
            ]
            result = subprocess.run(alternate, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(
                (data / "launches/alternate-evaluator.json").read_text()
            )
            self.assertIn("--gres=gpu:rtx_pro_6000:2", receipt["training"])
            self.assertIn("--gres=gpu:a5000:1", receipt["validation"])
            self.assertEqual(receipt["settings"]["OLMO_ROOT"], str(evaluation))
            self.assertEqual(receipt["settings"]["OLMO_DATA"], str(data))
            sharded = alternate.copy()
            sharded[sharded.index("--name") + 1] = "parallel-evaluator"
            shards = root / "shards"
            shards.mkdir()
            (shards / "part000.h5").touch()
            (shards / "manifest.json").write_text(
                json.dumps(
                    {
                        "full_release": True,
                        "start": 0,
                        "stop": 16,
                        "source_rows": 16,
                        "batch_size": 16,
                        "source": str(evaluation / "eval/lfmc/inputs.h5"),
                        "shards": [{"file": "part000.h5"}],
                    }
                )
            )
            (evaluation / "execution-context.json").write_text("{}")
            rejected = subprocess.run(sharded, env=env, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            sharded += ["--shard-inputs", str(shards)]
            result = subprocess.run(sharded, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(
                (data / "launches/parallel-evaluator.json").read_text()
            )
            self.assertIn("dispatch_job", receipt)
            self.assertNotIn("validation_job", receipt)
            self.assertFalse(any(a.startswith("--gres") for a in receipt["validation"]))
            self.assertTrue(
                receipt["validation"][-1].endswith("dispatch_sharded_eval.sbatch")
            )
            self.assertEqual(receipt["settings"]["OLMO_SHARD_INPUTS"], str(shards))
            self.assertNotIn("OLMO_INFERENCE_HASHSEED", receipt["settings"])


if __name__ == "__main__":
    unittest.main()
