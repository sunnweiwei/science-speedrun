import unittest
import numpy as np
from ffn_agglomeration import score_pair, accept_pair, accept_pathfinder_pair


class ResegmentationScoringTest(unittest.TestCase):
    def test_pathfinder_rule_is_not_the_manual_starting_rule(self):
        s=dict(iou=.85,consistency=[[.7]*2]*2,prediction_voxels=[100,100],deleted_voxels=[0,0])
        self.assertTrue(accept_pair(s,complete=True))
        self.assertFalse(accept_pathfinder_pair(s,complete=True))
        s.update(iou=.95,consistency=[[.95]*2]*2,deleted_voxels=[100,100])
        self.assertFalse(accept_pair(s,complete=True))
        self.assertTrue(accept_pathfinder_pair(s,complete=True))

    def test_pathfinder_strict_boundaries_and_empty_masks(self):
        s=dict(iou=.85,consistency=[[.85]*2]*2,prediction_voxels=[100,100],deleted_voxels=[1,100])
        self.assertTrue(accept_pathfinder_pair(s,complete=True))
        s['deleted_voxels']=[2,2]
        self.assertFalse(accept_pathfinder_pair(s,complete=True))
        s.update(iou=.9,consistency=[[1.]*2]*2)
        self.assertFalse(accept_pathfinder_pair(s,complete=True))
        s.update(iou=.95,consistency=[[.9]*2]*2)
        self.assertFalse(accept_pathfinder_pair(s,complete=True))
        s.update(consistency=[[1.]*2]*2,prediction_voxels=[0,100])
        self.assertFalse(accept_pathfinder_pair(s,complete=True))
        s['prediction_voxels']=[100,100];s['deleted_voxels']=[-1,0]
        with self.assertRaises(ValueError):accept_pathfinder_pair(s,complete=True)
        self.assertFalse(accept_pathfinder_pair({},complete=False))

    def setUp(self):
        self.base = np.array([[[1, 1, 2, 2]]], dtype=np.uint32)

    def test_two_seeds_recover_union(self):
        both = np.ones_like(self.base, dtype=bool)
        score = score_pair(self.base, [1, 2], both, both, 0, 1)
        self.assertTrue(accept_pair(score, complete=True))
        self.assertFalse(accept_pair(score, complete=False))
        self.assertEqual(score['consistency'], [[1., 1.], [1., 1.]])

    def test_unconnected_fragments_rejected(self):
        score = score_pair(self.base, [1, 2], self.base == 1, self.base == 2, 0, 0)
        self.assertFalse(accept_pair(score, complete=True))

    def test_strict_threshold_and_all_consistencies(self):
        both = np.ones_like(self.base, dtype=bool)
        score = score_pair(self.base, [1, 2], both, both, 0, 0)
        score['iou'] = .8
        self.assertFalse(accept_pair(score, complete=True))
        score['iou'] = 1.
        score['consistency'][1][0] = .6
        self.assertFalse(accept_pair(score, complete=True))

    def test_both_unstable_masks_rejected(self):
        both = np.ones_like(self.base, dtype=bool)
        score = score_pair(self.base, [1, 2], both, both, 1, 1)
        self.assertFalse(accept_pair(score, complete=True))


if __name__ == '__main__':
    unittest.main()
