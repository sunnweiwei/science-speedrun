import unittest
import numpy as np
from audit_training_overlap import node_exposure


class ExposureTests(unittest.TestCase):
    def test_outer_corner_half_open_bounds_and_mask(self):
        mask=np.ones((2,2,2),bool);mask[1,1,1]=False
        points=[[10,20,30],[11,22,33],[12,24,36],[14,20,30],[9.99,20,30]]
        inside,valid=node_exposure(points,[10,20,30],[2,4,6],mask)
        np.testing.assert_array_equal(inside,[True,True,True,False,False])
        np.testing.assert_array_equal(valid,[True,True,False,False,False])

    def test_invalid_geometry_and_nonboolean_mask_rejected(self):
        for spacing,point,mask in [([0,1,1],[0,0,0],np.ones((2,2,2),bool)),
            ([1,1,1],[float('nan'),0,0],np.ones((2,2,2),bool)),
            ([1,1,1],[0,0,0],np.ones((2,2,2),np.uint8))]:
            with self.assertRaises(ValueError):node_exposure([point],[0,0,0],spacing,mask)


if __name__=='__main__':unittest.main()
