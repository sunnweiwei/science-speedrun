import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/campaign'))
from summarize_shared import TRACKS, TEST_COUNTS, checked_run
from summarize_full import package_hash, checked_score, TEST_SUBJECT_COUNTS


class FullReceiptTests(unittest.TestCase):
    def test_smoke_and_subset_rejected(self):
        self.assertEqual(TRACKS, ('tuev', 'ad65', 'openbmi', 'longitudinal'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'eegnet-s0'
            pkg = run / 'submission'
            score = run / 'score'
            pkg.mkdir(parents=True)
            score.mkdir()
            (pkg / 'infer.sh').write_text('exit 0\n')
            identity = {t: 'a' * 64 for t in TRACKS}
            campaign = {'development_identity': {}, 'test_data_identity': identity,
                        'test_subject_counts': {t: 1 for t in TRACKS}}
            (run / 'submission-full-run.json').write_text(json.dumps(
                {'status': 'complete', 'completed_epochs': 50, 'model': 'eegnet', 'seed': 0,
                 'best_epoch': 12, 'best_validation_mean': .4,
                 'pipeline_version': 'shared-v2.1-channel-preserving',
                 'training_coverage': {t: {'complete_passes': 1} for t in TRACKS}}))
            (pkg / 'config.json').write_text(json.dumps({'schema': 'bci-shared-package-v2', 'family': 'eegnet',
                                                        'pipeline_version': 'shared-v2.1-channel-preserving'}))
            (pkg / 'training.json').write_text(json.dumps(
                {'model': 'eegnet', 'seed': 0, 'scope': 'full', 'data_identity': {},
                 'best_epoch': 12, 'best_validation_mean': .4,
                 'pipeline_version': 'shared-v2.1-channel-preserving'}))
            digest = package_hash(pkg)
            details = {t: {'status': 'ok', 'reward': .5, 'sample_count': TEST_COUNTS[t],
                           'subject_count': 1, 'data_identity': identity[t],
                           'package_sha256': digest} for t in TRACKS}
            report = {'scope': 'full', 'tracks': details, 'package_sha256': digest}
            (score / 'report.json').write_text(json.dumps(report))
            (score / 'reward.json').write_text(json.dumps({'score': .1625, 'reward': .1625, **{t: .5 for t in TRACKS}}))
            (run / 'score.exit').write_text('0\n')
            self.assertEqual(checked_run(root, campaign, 'eegnet', 0)['tuev'], .5)
            with self.assertRaises(ValueError):
                checked_run(root, campaign, 'eegnet', 0, expected_epochs=100)
            receipt_path = run / 'submission-full-run.json'
            duration_receipt = json.loads(receipt_path.read_text())
            duration_receipt['completed_epochs'] = 100
            receipt_path.write_text(json.dumps(duration_receipt))
            self.assertEqual(checked_run(root, campaign, 'eegnet', 0, expected_epochs=100)['tuev'], .5)
            with self.assertRaises(ValueError):
                checked_run(root, campaign, 'eegnet', 0)
            duration_receipt['completed_epochs'] = 50
            receipt_path.write_text(json.dumps(duration_receipt))
            report['scope'] = 'smoke'
            (score / 'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                checked_run(root, campaign, 'eegnet', 0)
            report['scope'] = 'full'
            for field, bad in (('sample_count', 1), ('subject_count', 2),
                               ('data_identity', 'b' * 64), ('package_sha256', 'b' * 64)):
                original = details['tuev'][field]
                details['tuev'][field] = bad
                (score / 'report.json').write_text(json.dumps(report))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    checked_run(root, campaign, 'eegnet', 0)
                details['tuev'][field] = original
            (score / 'report.json').write_text(json.dumps(report))
            (pkg / 'infer.sh').write_text('exit 1\n')
            with self.assertRaises(ValueError):
                checked_run(root, campaign, 'eegnet', 0)

    def test_specialist_full_receipt_rejects_incomplete_or_unbound_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'eegnet-tuev-s0'
            pkg = run / 'submission/tuev'
            score = run / 'score'
            pkg.mkdir(parents=True)
            score.mkdir()
            receipt = {'status': 'complete', 'completed_epochs': 50,
                       'model': 'eegnet', 'track': 'tuev', 'seed': 0}
            receipt_path = run / 'submission/tuev-full-run.json'
            receipt_path.write_text(json.dumps(receipt))
            (pkg / 'config.json').write_text(json.dumps({'model': 'eegnet', 'track': 'tuev'}))
            (pkg / 'training.json').write_text(json.dumps({'seed': 0, 'scope': 'full-training'}))
            (pkg / 'weights.pt').write_bytes(b'fixture')
            digest = package_hash(pkg)
            detail = {'status': 'ok', 'reward': .5, 'sample_count': TEST_COUNTS['tuev'],
                      'subject_count': TEST_SUBJECT_COUNTS['tuev'],
                      'data_identity': 'a' * 64, 'package_sha256': digest}
            report = {'scope': 'full', 'package_sha256': digest, 'tracks': {'tuev': detail}}
            (score / 'reward.json').write_text(json.dumps({'tuev': .5}))
            (run / 'score.exit').write_text('0')
            jobs = {'eegnet-tuev-s0': {'status': 'SUCCEED', 'verified': True,
                                       'test_data_identity': 'a' * 64}}

            def evaluate():
                (score / 'report.json').write_text(json.dumps(report))
                return checked_score(root, jobs, 'eegnet', 'tuev', 0)

            self.assertEqual(evaluate(), .5)
            for obj, key, bad in [(report, 'scope', 'smoke'),
                                  (detail, 'sample_count', 1),
                                  (detail, 'subject_count', 2),
                                  (detail, 'subject_count', True),
                                  (detail, 'data_identity', 'b' * 64),
                                  (detail, 'package_sha256', 'b' * 64),
                                  (report, 'package_sha256', 'b' * 64)]:
                old = obj[key]
                obj[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    evaluate()
                obj[key] = old
            for key, bad in [('model', 'biot'), ('track', 'ad65'), ('seed', 1)]:
                old = receipt[key]
                receipt[key] = bad
                receipt_path.write_text(json.dumps(receipt))
                with self.subTest(identity=key), self.assertRaises(ValueError):
                    evaluate()
                receipt[key] = old
            receipt_path.write_text(json.dumps(receipt))
            (pkg / 'weights.pt').write_bytes(b'changed package')
            with self.assertRaises(ValueError):
                evaluate()


if __name__ == '__main__':
    unittest.main()
