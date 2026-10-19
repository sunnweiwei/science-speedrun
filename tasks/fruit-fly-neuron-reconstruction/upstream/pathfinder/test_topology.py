import unittest
import numpy as np
from topology import counts, sample, interior_segments

class TopologyTests(unittest.TestCase):
    def test_support_thresholds_and_omission(self):
        self.assertEqual(counts([1]*10, [11]*5+[12]*5, {11,12})['split_errors'], 1)
        self.assertEqual(counts([1]*9, [11]*5+[12]*4, {11,12})['split_errors'], 0)
        self.assertEqual(counts([1]*10+[2]*10, [11]*20, {11})['merge_errors'], 1)
        self.assertEqual(counts([1]*10+[2]*9, [11]*19, {11})['merge_errors'], 0)
        r = counts([1]*10, [0]*10, set())
        self.assertEqual(r['omitted_node_fraction'], 1)
        self.assertIsNone(r['nerl']); self.assertFalse(r['paper_protocol_verified'])

    def test_boundary_does_not_suppress_merges(self):
        r = counts([1]*10+[2]*10, [11]*20, set())
        self.assertEqual(r['merge_errors'], 1)
        self.assertEqual(counts([1]*10, [11]*5+[12]*5, {11})['split_errors'], 0)

    def test_large_ids_and_coordinates(self):
        seg = np.full((3,4,5), 2**63+9, np.uint64)
        sampled = sample(seg, [[100,208,316]], [100,200,300], [40,4,4])
        self.assertEqual(int(sampled[0]), 2**63+9)
        self.assertEqual(interior_segments(seg, [1,1,1]), {2**63+9})
        with self.assertRaises(ValueError): sample(seg, [[99,208,316]], [100,200,300], [40,4,4])
        with self.assertRaises(ValueError): interior_segments(seg, [2,1,1])
        with self.assertRaises(ValueError): counts([1], [1.5], {1})

if __name__ == '__main__': unittest.main()
