import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
from bci_score import AGGREGATE_TRACKS, CHANCE, CLASSES, aggregate, chance_adjusted, composite_score


class AggregateTests(unittest.TestCase):
    def test_chance_is_zero_and_perfect_is_one(self):
        self.assertEqual(aggregate({t: CHANCE[t] for t in CLASSES}), 0.0)
        self.assertEqual(aggregate({t: 1.0 for t in CLASSES}), 1.0)
        self.assertEqual(composite_score({t: CHANCE[t] for t in CLASSES})['score'], 0.0)
        self.assertEqual(composite_score({t: 1.0 for t in CLASSES})['score'], 1.0)

    def test_faced_removed_from_v4_scoring(self):
        self.assertEqual(tuple(CLASSES), ('tuev', 'ad65', 'openbmi', 'longitudinal'))
        self.assertEqual(AGGREGATE_TRACKS, ('tuev', 'ad65', 'openbmi', 'longitudinal'))
        self.assertNotIn('faced', CLASSES)
        with self.assertRaises(ValueError):
            chance_adjusted('faced', 0.5)

    def test_below_chance_stays_negative_in_signed_and_clips_to_zero_in_score(self):
        self.assertLess(chance_adjusted('tuev', 0.0), 0.0)
        zeros = {t: 0.0 for t in CLASSES}
        comp = composite_score(zeros)
        self.assertLess(comp['signed_score'], 0.0)
        self.assertEqual(comp['score'], 0.0)
        self.assertEqual(comp['reward'], 0.0)
        self.assertIsNone(aggregate({t: 0.6 for t in CLASSES if t != 'ad65'}))
        self.assertIsNone(aggregate({**{t: 0.6 for t in CLASSES}, 'openbmi': float('nan')}))


if __name__ == '__main__':
    unittest.main()
