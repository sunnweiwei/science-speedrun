import unittest
import numpy as np
from prepare_sparse_evaluation import center_bounds


class SparseGeometryTests(unittest.TestCase):
    def test_half_open_center_bounds(self):
        start,stop=center_bounds([40,32,32],[200,160,160],[80,64,64],[10]*3)
        np.testing.assert_array_equal(start,[0]*3)
        np.testing.assert_array_equal(stop,[2]*3)
        # Exact upper-face voxel center is excluded; exact lower center included.
        centers=(np.arange(start[0],stop[0])+.5)*80
        np.testing.assert_array_equal(centers,[40,120])

    def test_prediction_grid_and_native_grid_need_not_align(self):
        start,stop=center_bounds([3256,4640,4640],[26744,28128,28128],[80,64,64],[375,512,512])
        for lo,hi,a,b,s in zip([3256,4640,4640],[26744,28128,28128],start,stop,[80,64,64]):
            centers=(np.arange(a,b)+.5)*s
            self.assertTrue(((centers>=lo)&(centers<hi)).all())
            self.assertLess((a-.5)*s,lo)
            self.assertGreaterEqual((b+.5)*s,hi)

    def test_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError):center_bounds([-1,0,0],[10]*3,[1]*3,[20]*3)


if __name__=='__main__':unittest.main()
