import copy
import unittest

from compare_trajectories import compare


class TrajectoryComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(step=i, reward=-1., observation=[1., 2.],
                          normalized_action=[.1]) for i in (1, 2)]

    def test_equal_and_changed_values(self):
        self.assertTrue(compare(self.rows, self.rows, 2)['exact_numeric_equality'])
        other = copy.deepcopy(self.rows)
        other[1]['observation'][0] += .25
        result = compare(self.rows, other, 2)
        self.assertFalse(result['exact_numeric_equality'])
        self.assertEqual(result['fields']['observation']['max_abs_error'], .25)
        self.assertEqual(result['fields']['observation']['first_exact_difference_step'], 2)
        self.assertEqual(result['left_raw_return'], -2.)

    def test_rejects_short_or_reordered_episodes(self):
        for rows in (self.rows[:1], self.rows[::-1], self.rows * 2):
            with self.assertRaises(ValueError):
                compare(self.rows, rows, 2)

    def test_requires_explicit_common_field_subset(self):
        other = copy.deepcopy(self.rows)
        for row in other:
            del row['observation']
        with self.assertRaises(ValueError):
            compare(self.rows, other, 2)
        self.assertTrue(compare(self.rows, other, 2, ['reward'])['exact_numeric_equality'])

    def test_rejects_nonfinite_or_different_dimension(self):
        for value in ([1.], [[1., 2.]], [float('nan'), 2.], [True, 2.]):
            other = copy.deepcopy(self.rows)
            other[0]['observation'] = value
            with self.assertRaises(ValueError):
                compare(self.rows, other, 2)


if __name__ == '__main__':
    unittest.main()
