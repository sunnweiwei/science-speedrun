import unittest
import numpy as np
from prepare_malecns_training import supervision, sample_centers


class HeldOutSupervisionTest(unittest.TestCase):
    def test_shared_ids_are_unknown_not_background_negatives(self):
        labels = np.array([[[0, 11, 12, 13]]], dtype=np.uint64)
        np.testing.assert_array_equal(supervision(labels, {12}), [[[False, True, False, True]]])
        self.assertEqual(int(labels[0,0,2]), 12)

    def test_centers_avoid_excluded_ids_and_keep_real_context(self):
        labels = np.ones((144,)*3, dtype=np.uint16)
        labels[:72] = 2
        centers = sample_centers(labels, supervision(labels, {2}))
        self.assertTrue((centers[:,0] >= 72).all())
        self.assertTrue((centers >= 64).all())
        self.assertTrue((centers < 80).all())


if __name__ == '__main__':
    unittest.main()
