import unittest
import numpy as np
from inspect_fib25 import map_bodies


class BodyMappingTest(unittest.TestCase):
    def test_unsorted_many_to_one(self):
        stack = np.array([[[0, 4, 8, 4]]], dtype=np.int32)
        pairs = np.array([[8, 99], [0, 0], [4, 99]], dtype=np.int64)
        np.testing.assert_array_equal(map_bodies(stack, pairs), [[[0, 99, 99, 99]]])

    def test_missing_or_duplicate_mapping(self):
        stack = np.array([[[8]]])
        for pairs in (np.array([[0, 0]]), np.array([[8, 2], [8, 3]])):
            with self.assertRaises(ValueError):
                map_bodies(stack, pairs)

    def test_background_is_preserved(self):
        with self.assertRaises(ValueError):
            map_bodies(np.array([[[0]]]), np.array([[0, 5]]))


if __name__ == '__main__':
    unittest.main()
