import json
from pathlib import Path
import tempfile
import unittest

from stage1_status import digest, snapshot


class StatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train, self.evaluation = self.root / 'train', self.root / 'evaluation'
        self.train.mkdir()
        self.evaluation.mkdir()
        self.spec = dict(case='case', planned_controls=100000, training_run='train',
                         evaluation_dir='evaluation', checkpoint='train/candidate.pt')
        self.write(self.train / 'episodes.jsonl', dict(phase='training', steps=400))
        self.write(self.train / 'config.json', dict(algorithm='PPO', seed=42,
                   recipe='public', training_steps=400))
        self.write(self.train / 'completed.json', {})
        (self.train / 'candidate.pt').write_bytes(b'test-only')
        self.write(self.evaluation / 'protocol.json', dict(environment='case', horizon=200))
        self.write(self.evaluation / 'completed.json', dict(returns=[-5.],
            checkpoint_sha256=digest(self.train / 'candidate.pt')))
        self.row = dict(phase='evaluation', training_step=400, steps=200,
                       mean_drag=1., return_value=-5., rms_lift=.3, last_half_mean_drag=.9)
        self.row['return'] = self.row.pop('return_value')
        self.write(self.evaluation / 'episodes.jsonl', self.row)
        self.write(self.evaluation / 'evaluation-steps.jsonl', {})

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + '\n')

    def test_pilot_completion_is_not_full_budget_or_paper_match(self):
        result = snapshot(self.root, self.spec)
        self.assertEqual(result['budget_progress_percent'], .4)
        self.assertTrue(result['run_budget_completed'])
        self.assertFalse(result['planned_budget_completed'])
        self.assertFalse(result['paper_match_accepted'])
        self.assertIsNone(result['paper_gap'])

    def test_changed_checkpoint_rejected(self):
        (self.train / 'candidate.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            snapshot(self.root, self.spec)

    def test_short_episode_rejected(self):
        self.row['steps'] = 20
        self.write(self.evaluation / 'episodes.jsonl', self.row)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            snapshot(self.root, self.spec)

    def test_wrong_case_rejected(self):
        self.spec['case'] = 'different'
        with self.assertRaisesRegex(ValueError, 'case mismatch'):
            snapshot(self.root, self.spec)

    def test_pending_evaluation_has_no_score(self):
        self.spec['evaluation_dir'] = 'pending'
        self.assertEqual(snapshot(self.root, self.spec)['independent_evaluations'], [])


if __name__ == '__main__':
    unittest.main()
