import unittest
import numpy as np
from evaluate_fib25 import voxel_metrics


class FibScoreTests(unittest.TestCase):
    def test_label_permutation_and_large_ids(self):
        gt=np.array([0,1,1,2,2,3,3,3],dtype=np.uint64).reshape(2,2,2)
        pred=np.array([99,2**40,2**40,5,5,7,7,7],dtype=np.uint64).reshape(2,2,2)
        m=voxel_metrics(gt,pred)
        self.assertEqual(m['vi_total_bits'],0)
        self.assertEqual(m['adapted_rand_error'],0)

    def test_prediction_omissions_remain_in_denominator(self):
        gt=np.array([1,1,1,1,2,2,2,2],dtype=np.uint32).reshape(2,2,2)
        m=voxel_metrics(gt,np.zeros_like(gt))
        self.assertEqual(m['scored_voxels'],8)
        self.assertEqual(m['omitted_prediction_fraction'],1)
        self.assertGreater(m['vi_merge_bits'],0)
        self.assertGreater(m['adapted_rand_error'],0)


if __name__=='__main__':unittest.main()
