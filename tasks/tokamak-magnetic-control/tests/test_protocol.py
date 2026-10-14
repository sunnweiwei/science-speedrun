"""Protocol failure handling with synthetic policies (no simulator)."""

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "environment" / "evaluation", Path("/opt/tokamak-task/evaluation")):
    sys.path.insert(0, str(_p))
import protocol  # noqa: E402

GOOD = """import json,sys
for line in sys.stdin:
    r=json.loads(line); out={"request_id":r["request_id"]}
    if r["op"]=="reset": out["ready"]=True
    elif r["op"]=="act": out["action"]=[0.0]*len(r["observation"][:12]) if False else [0.0]*12
    else: out["closed"]=True
    print(json.dumps(out),flush=True)
    if r["op"]=="close": break
"""


def make_package(body):
    d = Path(tempfile.mkdtemp())
    (d / "policy.py").write_text(body)
    (d / "infer.sh").write_text("#!/usr/bin/env bash\ncd \"$(dirname \"$0\")\"\nexec python3 policy.py\n")
    os.chmod(d / "infer.sh", os.stat(d / "infer.sh").st_mode | stat.S_IEXEC)
    return d


class ProtocolTests(unittest.TestCase):
    def run_policy(self, body, action=None):
        pkg = make_package(body)
        pol = protocol.Policy(pkg, pkg / "stderr.txt", timeout=10)
        pol.reset({"environment": "x", "family": "y", "seed": 0}, [585], [12], [-1.0] * 12, [1.0] * 12)
        return pol

    def test_good_policy_round_trip(self):
        pol = self.run_policy(GOOD)
        self.assertEqual(pol.act([0.0] * 585), [0.0] * 12)
        self.assertGreater(len(pol.latencies), 0)
        pol.close()

    def test_out_of_bounds_action_is_policy_error(self):
        pol = self.run_policy(GOOD.replace("[0.0]*12", "[5.0]*12"))
        with self.assertRaises(protocol.PolicyError):
            pol.act([0.0] * 585)
        pol.abort()

    def test_wrong_size_action_is_policy_error(self):
        pol = self.run_policy(GOOD.replace("[0.0]*12", "[0.0]*11"))
        with self.assertRaises(protocol.PolicyError):
            pol.act([0.0] * 585)
        pol.abort()

    def test_nan_action_is_policy_error(self):
        pol = self.run_policy(GOOD.replace("[0.0]*12", "[float('nan')]*12"))
        with self.assertRaises(protocol.PolicyError):
            pol.act([0.0] * 585)
        pol.abort()

    def test_crashing_policy_is_policy_error(self):
        pol = self.run_policy(GOOD.replace('elif r["op"]=="act":', 'elif r["op"]=="act": sys.exit(3)\n    elif False:'))
        with self.assertRaises(protocol.PolicyError):
            pol.act([0.0] * 585)
        pol.abort()

    def test_bad_observation_is_infrastructure_error(self):
        pol = self.run_policy(GOOD)
        with self.assertRaises(protocol.InfrastructureError):
            pol.act([0.0] * 10)
        pol.abort()


if __name__ == "__main__":
    unittest.main()
