"""A root-run evaluator must drop the submitted policy to an unprivileged, short-lived process."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ISOLATED = {'HYDROGYM_POLICY_ISOLATION': '1'}

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
import protocol
from protocol import InfrastructureError, Policy

POLICY = r'''
import json, os, subprocess, sys
target = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "target.txt")).read().strip()
try:
    open(target, "w").write("forged")
except OSError as error:
    sys.stderr.write("write-denied %s\n" % type(error).__name__)
sys.stderr.write("uid=%d\n" % os.getuid()); sys.stderr.flush()
subprocess.Popen(["setsid", "sleep", "300"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for line in sys.stdin:
    request = json.loads(line)
    reply = {"protocol": request["protocol"], "request_id": request["request_id"]}
    reply.update({"reset": {"ready": True}, "act": {"action": [0.0]}, "close": {"closed": True}}[request["op"]])
    print(json.dumps(reply), flush=True)
    if request["op"] == "close":
        break
'''


def account_exists():
    try:
        import pwd
        pwd.getpwuid(protocol.POLICY_UID)
        return True
    except KeyError:
        return False


class IsolationSwitchTests(unittest.TestCase):
    def test_isolation_is_off_unless_requested(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HYDROGYM_POLICY_ISOLATION', None)
            self.assertIsNone(protocol.policy_account())

    @unittest.skipIf(os.geteuid() == 0, 'checks the non-root refusal')
    def test_requested_isolation_refuses_non_root(self):
        with patch.dict(os.environ, ISOLATED), self.assertRaises(InfrastructureError):
            protocol.policy_account()

    def test_verifier_requests_isolation_offline(self):
        here = Path(__file__).resolve().parents[1]
        self.assertIn('export HYDROGYM_POLICY_ISOLATION=1', (here / 'tests/test.sh').read_text())
        self.assertIn('network_mode: none', (here / 'tests/docker-compose.yaml').read_text())


@unittest.skipUnless(os.geteuid() == 0 and account_exists(), 'needs root and the policy account (task image)')
class PolicyIsolationTests(unittest.TestCase):
    def test_policy_runs_unprivileged_cannot_write_results_and_leaves_no_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            os.chmod(root, 0o755)
            package = root / 'package'; package.mkdir(); os.chmod(package, 0o755)
            (package / 'policy.py').write_text(POLICY)
            (package / 'infer.sh').write_text('exec python3 "$(dirname "$0")/policy.py"\n')
            results = root / 'results'; results.mkdir(mode=0o755)
            forged = results / 'reward.json'
            (package / 'target.txt').write_text(str(forged))
            for path in package.iterdir():
                path.chmod(0o444)
            with patch.dict(os.environ, ISOLATED):
                policy = Policy(package, root / 'policy.log', timeout=20)
            policy.reset({'environment': 'x', 'family': 'x', 'seed': 1}, [1], [1], [-1.0], [1.0])
            self.assertEqual(policy.act([0.5]), [0.0])
            policy.close()
            log = (root / 'policy.log').read_text()
            self.assertIn(f'uid={protocol.POLICY_UID}', log)
            self.assertIn('write-denied PermissionError', log)
            self.assertFalse(forged.exists())
            time.sleep(0.5)
            def alive(proc):  # killed children re-parented to a non-reaping PID 1 stay as zombies
                try:
                    return proc.stat().st_uid == protocol.POLICY_UID and \
                        (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
                except FileNotFoundError:
                    return False
            owners = [p for p in Path('/proc').iterdir() if p.name.isdigit() and alive(p)]
            self.assertEqual(owners, [])

    def test_missing_account_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(protocol, 'POLICY_UID', 54321), \
                patch.dict(os.environ, ISOLATED):
            with self.assertRaises(InfrastructureError):
                Policy(Path(tmp), Path(tmp) / 'policy.log', timeout=5)


if __name__ == '__main__':
    unittest.main()
