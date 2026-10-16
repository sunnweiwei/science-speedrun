"""Archive portability tests; synthetic fixtures are not CFD evidence."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from archive_results import FILES, archive, validate_final_metrics


class ArchiveTests(unittest.TestCase):
    def test_nonfinite_metrics_cannot_pass_replay_comparison(self):
        for key in ('return', 'mean_drag', 'rms_lift'):
            for value in (float('nan'), float('inf'), float('-inf'), True, '-1'):
                for side in (0, 1):
                    with self.subTest(key=key, value=value, side=side):
                        rows = [dict(zip(('return', 'mean_drag', 'rms_lift'), (-123., 1., .2))) for _ in range(2)]
                        rows[side][key] = value
                        with self.assertRaisesRegex(ValueError, 'Nonfinite or invalid'):
                            validate_final_metrics(*rows)

    def test_finite_replay_tolerance_is_unchanged(self):
        validate_final_metrics({'return': -10.}, {'return': -10. + 1e-6})
        with self.assertRaisesRegex(ValueError, 'replay disagrees'):
            validate_final_metrics({'return': -10.}, {'return': -10. + 1e-4})

    def test_cavity_and_recipe_sources_survive_archiving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'cavity'
            run.mkdir()
            for name in FILES:
                (run / name).write_text('{}\n')
            (run / 'config.json').write_text(json.dumps(dict(algorithm='PPO', seed=42,
                recipe='ppo-scaled', training_steps=100)))
            checksum = hashlib.sha256((run / 'candidate.pt').read_bytes()).hexdigest()
            (run / 'workflow-completed.json').write_text(json.dumps(dict(
                status='cavity-trained-and-replayed-not-paper-matched', policy_sha256=[checksum])))
            training = dict(phase='training', steps=100)
            evaluation = dict(phase='evaluation', steps=100, training_step=100, **{'return': -123.})
            (run / 'episodes.jsonl').write_text(json.dumps(training)+'\n'+json.dumps(evaluation)+'\n')
            independent = run / 'independent-final'
            independent.mkdir()
            for name in ('completed.json', 'protocol.json', 'evaluation-steps.jsonl'):
                (independent / name).write_text('{}\n')
            (independent / 'episodes.jsonl').write_text(json.dumps(evaluation)+'\n')
            source = run / 'source'
            (source / 'recipes').mkdir(parents=True)
            for name, value in [('learner.py', '# source\n'), ('run.sh', '#!/bin/bash\n'),
                                ('Dockerfile', 'FROM fixture\n'), ('recipes/ppo-scaled.json', '{"reward_scale": 0.1}\n')]:
                (source / name).write_text(value)
            (source / '__pycache__').mkdir()
            (source / '__pycache__/ignored.pyc').write_bytes(b'cache')
            output = root / 'archive'
            result = archive([run], output)
            self.assertEqual(result['runs'], 1)
            manifest = json.loads((output / 'manifest.json').read_text())
            for name in ('learner.py', 'run.sh', 'Dockerfile', 'recipes/ppo-scaled.json'):
                relative = 'cavity/source/' + name
                self.assertEqual((output / relative).read_bytes(), (source / name).read_bytes())
                self.assertIn(relative, manifest['sha256'])
            self.assertFalse((output / 'cavity/source/__pycache__').exists())
            with self.assertRaisesRegex(ValueError, 'destination exists'):
                archive([run], output)


if __name__ == '__main__':
    unittest.main()
