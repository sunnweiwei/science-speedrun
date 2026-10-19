import unittest
import numpy as np
from ffn_seeds import policy_peaks,seed_is_clear

class SeedTests(unittest.TestCase):
    def test_flat_image_has_no_finite_interior_distance_peaks(self):
        self.assertEqual(policy_peaks(np.zeros((48,)*3,np.uint8)).shape,(0,3))

    def test_seed_order_and_exclusion_are_deterministic(self):
        image=np.random.RandomState(7).randint(0,256,(48,)*3).astype(np.uint8)
        seeds=policy_peaks(image)
        self.assertGreater(len(seeds),0)
        np.testing.assert_array_equal(seeds,policy_peaks(image))
        np.testing.assert_array_equal(seeds,np.asarray(sorted(map(tuple,seeds))))
        self.assertTrue(((seeds>=16)&(seeds<32)).all())
        mask=np.zeros(image.shape,bool);mask[16:24,:,:]=True
        self.assertTrue((policy_peaks(image,exclusion=mask)[:,0]>=24).all())
        self.assertEqual(policy_peaks(image,exclusion=np.ones(image.shape,bool)).shape,(0,3))

    def test_seed_near_other_segment_is_excluded_but_image_edge_does_not_wrap(self):
        labels=np.zeros((5,)*3,np.uint32);labels[4,4,4]=1
        self.assertTrue(seed_is_clear(labels,(0,0,0)))
        self.assertFalse(seed_is_clear(labels,(3,3,3)))
        self.assertTrue(seed_is_clear(labels,(3,3,3),boundary_distance=0))
        self.assertFalse(seed_is_clear(labels,(4,4,4),boundary_distance=0))

if __name__=='__main__':unittest.main()
