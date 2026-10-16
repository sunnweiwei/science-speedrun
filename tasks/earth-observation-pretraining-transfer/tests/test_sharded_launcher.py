"""Check array dependencies, fixed inference hashing and CPU-only merge requests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ShardedLauncherTest(unittest.TestCase):
    def test_submission_and_context_guards(self):
        track = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "packages/model"
            package.mkdir(parents=True)
            for name in ["encoder.pt", "config.json", "infer.py", "LICENSE-OlmoEarth"]:
                (package / name).touch()
            (package / "provenance.json").write_text("{}")
            source = root / "eval/lfmc/inputs.h5"
            source.parent.mkdir(parents=True)
            source.touch()
            shards = root / "shards"
            shards.mkdir()
            (shards / "part000.h5").touch()
            (shards / "manifest.json").write_text(
                json.dumps(
                    {
                        "source": str(source),
                        "source_rows": 16,
                        "start": 0,
                        "stop": 16,
                        "batch_size": 16,
                        "full_release": True,
                        "shards": [{"file": "part000.h5", "start": 0, "stop": 16}],
                    }
                )
            )
            fake = root / "sbatch"
            fake.write_text(
                f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\np=Path(os.environ['FAKE_SLURM_LOG'])\na=json.loads(p.read_text()) if p.exists() else []\na.append({{'argv':sys.argv[1:], 'hash_seed':os.environ.get('OLMO_INFERENCE_HASHSEED')}})\np.write_text(json.dumps(a))\nprint(2000+len(a))\n"
            )
            fake.chmod(0o755)
            log = root / "calls.json"
            env = dict(
                os.environ, PATH=f"{root}:{os.environ['PATH']}", FAKE_SLURM_LOG=str(log)
            )
            command = [
                sys.executable,
                str(track / "docs/slurm/submit_sharded_eval.py"),
                "--root",
                str(root),
                "--name",
                "model",
                "--shard-inputs",
                str(shards),
                "--upstream",
                str(root / "runtime"),
                "--parsable",
            ]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "2003")
            calls = json.loads(log.read_text())
            self.assertEqual(len(calls), 3)
            for call in calls:
                self.assertIn("--partition=" + os.environ.get("EARTH_SLURM_PARTITION", "gpu"), call["argv"])
                self.assertEqual(call["hash_seed"], "0")
            self.assertIn("--gres=gpu:1", calls[0]["argv"])
            self.assertIn("--array=0-0", calls[0]["argv"])
            self.assertIn("--gres=gpu:1", calls[1]["argv"])
            self.assertFalse(any(v.startswith("--gres=") for v in calls[2]["argv"]))
            self.assertIn("--dependency=afterok:2001:2002", calls[2]["argv"])
            repeated = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(len(json.loads(log.read_text())), 3)


if __name__ == "__main__":
    unittest.main()
