import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
from aggregate import improvement, task_score

ROWS = {'a': {'primary_metric': {'id': 'drag', 'direction': 'lower_is_better'}, 'zero': 2.},
        'b': {'primary_metric': {'id': 'tke', 'direction': 'higher_is_better'}, 'zero': 1.}}


class AggregateTests(unittest.TestCase):
    def test_zero_control_scores_zero_and_tiny_wins_are_tiny(self):
        self.assertEqual(task_score({'a/drag': 2., 'b/tke': 1.}, ['a', 'b'], ROWS)['score'], 0.)
        tiny = task_score({'a/drag': 1.9999, 'b/tke': 1.0001}, ['a', 'b'], ROWS)
        self.assertLess(tiny['score'], 1e-3)

    def test_harm_and_failure_are_charged(self):
        result = task_score({'a/drag': 1., 'b/failed': 1}, ['a', 'b'], ROWS)
        self.assertEqual(result['per_case'], {'a': .5, 'b': -1.})
        self.assertEqual((result['raw_mean_improvement'], result['score']), (-.25, 0.))

    def test_clip(self):
        self.assertEqual(improvement(100., 1., 'higher_is_better'), 1.)
        self.assertEqual(improvement(100., 1., 'lower_is_better'), -1.)
        with self.assertRaises(ValueError): improvement(float('nan'), 1., 'lower_is_better')


if __name__ == '__main__': unittest.main()
