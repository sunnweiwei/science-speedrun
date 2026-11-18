import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_receipt', ROOT / 'tests/campaign/check_receipt.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
verifier_spec = importlib.util.spec_from_file_location('main_verifier', ROOT / 'tests/main_verifier.py')
verifier = importlib.util.module_from_spec(verifier_spec)
verifier_spec.loader.exec_module(verifier)


class ReceiptTests(unittest.TestCase):
    def test_flat_reward_and_report_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verifier.write_receipts(root, {'scope': 'smoke', 'package_sha256': 'a' * 64,
                                          'tracks': {'tuev': {'status': 'ok', 'reward': .5,
                                                              'package_sha256': 'a' * 64}}})
            self.assertEqual(module.checked_reward(root, 'tuev'), .5)
            for report in ({'scope': 'smoke', 'tracks': {'tuev': {'status': 'ok', 'reward': .5}}},
                           {'scope': 'smoke', 'package_sha256': 'a' * 64,
                            'tracks': {'tuev': {'status': 'invalid_submission', 'reward': .5,
                                                'package_sha256': 'a' * 64}}},
                           {'scope': 'smoke', 'package_sha256': 'a' * 64,
                            'tracks': {'tuev': {'status': 'ok', 'reward': .4,
                                                'package_sha256': 'a' * 64}}}):
                (root / 'report.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    module.checked_reward(root, 'tuev')

    def test_matrix_cli_and_mock_consumer_with_producer(self):
        mock_spec = importlib.util.spec_from_file_location('rsi_mock', ROOT / 'tests/rsi_mock.py')
        rsi_mock = importlib.util.module_from_spec(mock_spec)
        mock_spec.loader.exec_module(rsi_mock)
        tracks = rsi_mock.TRACKS
        self.assertEqual(tracks, ('tuev', 'ad65', 'openbmi', 'longitudinal'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = {'scope': 'smoke', 'package_sha256': 'a' * 64,
                      'tracks': {t: {'status': 'ok', 'reward': .5,
                                     'package_sha256': 'a' * 64} for t in tracks}}
            command = [sys.executable, str(ROOT / 'tests/campaign/check_receipt.py'),
                       str(root), '--tracks', *tracks]

            def check(ok):
                process = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(process.returncode == 0, ok)
                if ok:
                    self.assertEqual(rsi_mock.checked_rewards(root, tracks),
                                     {t: .5 for t in tracks})
                else:
                    with self.assertRaises(ValueError):
                        rsi_mock.checked_rewards(root, tracks)

            verifier.write_receipts(root, result)
            check(True)
            reward_path = root / 'reward.json'
            valid = json.loads(reward_path.read_text())
            for bad in ({**valid, 'tuev': .7},
                        {**valid, 'tuev': True},
                        {**valid, 'tuev': float('nan')},
                        {**valid, 'extra': .5},
                        {k: v for k, v in valid.items() if k != 'tuev'}):
                reward_path.write_text(json.dumps(bad))
                check(False)
            verifier.write_receipts(root, result)
            result['tracks']['tuev']['status'] = 'invalid_submission'
            verifier.write_receipts(root, result)
            check(False)


if __name__ == '__main__':
    unittest.main()
