import unittest
import numpy as np
from diagnose_tile_boundaries import diagnose


class TileBoundaryTest(unittest.TestCase):
    def truth(self):
        return dict(nodes=[dict(id=i, skeleton_id=1, position_nm=[0, 0, x])
                           for i, x in enumerate((9, 10, 19, 20, 21))],
                    edges=[[0, 1], [1, 2], [2, 3], [3, 4]])

    def test_physical_origin_boundary_and_omission(self):
        result = diagnose(self.truth(), np.array([1, 1, 2, 3, 0]), [0, 0, 10], [10]*3)
        cross = result['groups']['cross_tile']
        within = result['groups']['within_tile']
        self.assertEqual(cross['connected']['edges'], 1)
        self.assertEqual(cross['different_nonzero_labels']['edges'], 1)
        self.assertEqual(cross['split_fraction_among_assigned_edges'], .5)
        self.assertEqual(within['different_nonzero_labels']['length_nm'], 9)
        self.assertEqual(within['omitted_endpoint']['edges'], 1)
        self.assertEqual(within['split_fraction_among_assigned_edges'], 1)

    def test_no_assigned_edges_has_no_rate(self):
        result = diagnose(self.truth(), np.zeros(5, dtype=np.uint64), [0]*3, [10]*3)
        self.assertTrue(all(g['split_fraction_among_assigned_edges'] is None
                            for g in result['groups'].values()))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            diagnose(self.truth(), np.ones(5), [0]*3, [10]*3)
        with self.assertRaises(ValueError):
            diagnose(self.truth(), np.ones(5, dtype=int), [0]*3, [0]*3)
        truth = self.truth()
        truth['edges'].append([1, 0])
        with self.assertRaises(ValueError):
            diagnose(truth, np.ones(5, dtype=int), [0]*3, [10]*3)


if __name__ == '__main__':
    unittest.main()
