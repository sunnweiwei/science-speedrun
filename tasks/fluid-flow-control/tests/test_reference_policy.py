"""Contract regression tests for standalone exported reference actors."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "environment/evaluation"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires declared PyTorch runtime")
class ReferencePolicyTests(unittest.TestCase):
    def test_relocated_observation_conditioned_reference(self):
        import torch
        from protocol import Policy, PolicyError

        torch.set_num_threads(1)
        torch.manual_seed(900)
        model = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.Tanh()).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "original"
            package.mkdir()
            torch.jit.trace(model, torch.zeros(1, 5)).save(str(package / "model.pt"))
            cfg = {"observation_size": 5, "action_size": 3,
                   "model_sha256": hashlib.sha256((package / "model.pt").read_bytes()).hexdigest()}
            (package / "config.json").write_text(json.dumps(cfg))
            shutil.copyfile(TASK / "environment/baseline/starter/reference_predict.py", package / "predict.py")
            (package / "infer.sh").write_text('#!/bin/sh\nexec python "$(dirname "$0")/predict.py"\n')
            relocated = root / "relocated"
            shutil.move(package, relocated)
            policy = Policy(relocated, root / "stderr", timeout=20)
            try:
                for name in ("arbitrary-new-name", "different-order-and-name"):
                    policy.reset({"environment": name}, [5], [3], [-1.] * 3, [1.] * 3)
                    for obs in ([0.] * 5, [1., -.3, .5, -.1, 2.]):
                        actual = policy.act(obs)
                        expected = model(torch.tensor(obs).unsqueeze(0))[0].detach()
                        torch.testing.assert_close(torch.tensor(actual), expected)
                policy.close()
            finally:
                policy.abort()
            wrong = Policy(relocated, root / "wrong-stderr", timeout=20)
            try:
                with self.assertRaises(PolicyError):
                    wrong.reset({"environment": "same-name"}, [6], [3], [-1.] * 3, [1.] * 3)
            finally:
                wrong.abort()


if __name__ == "__main__":
    unittest.main()
