import unittest
import numpy as np
from instance_index import build_index


class IndexTests(unittest.TestCase):
    def test_disconnected_high_ids_cross_slab_and_real_center(self):
        a=np.zeros((9,8,7),np.uint32);a[1:8,2,3]=4000000000;a[0,0,0]=17;a[8,7,6]=17
        rows=build_index(a,2)
        self.assertEqual([r['id'] for r in rows],[17,4000000000])
        for r in rows:
            coords=np.argwhere(a==r['id'])
            self.assertEqual(r['size'],len(coords))
            self.assertEqual(r['bbox_zyx'],np.stack((coords.min(0),coords.max(0)+1),1).tolist())
            self.assertEqual(a[tuple(r['center_zyx'])],r['id'])
    def test_empty_and_slab_invariance(self):
        self.assertEqual(build_index(np.zeros((2,2,2),np.uint32)),[])
        a=np.arange(120,dtype=np.uint32).reshape(5,6,4)%7
        self.assertEqual(build_index(a,1),build_index(a,5))

    def test_streamed_centers_match_full_box_oracle_including_ties(self):
        rng=np.random.default_rng(73)
        for a in [np.ones((10,8,6),np.uint32),rng.integers(0,8,(15,12,9),dtype=np.uint32)]:
            expected=[]
            for sid in np.unique(a):
                if not sid:continue
                xyz=np.argwhere(a==sid);center=xyz[np.argmin(((xyz-xyz.mean(0))**2).sum(1))]
                expected.append(dict(id=int(sid),size=len(xyz),bbox_zyx=np.stack((xyz.min(0),xyz.max(0)+1),1).tolist(),center_zyx=center.tolist()))
            for depth in (1,2,7,20):self.assertEqual(build_index(a,depth,100),expected)

    def test_global_instance_bound_not_only_per_slab(self):
        a=np.arange(1,10,dtype=np.uint32).reshape(9,1,1)
        with self.assertRaisesRegex(ValueError,'Instance limit'):build_index(a,1,8)
        with self.assertRaisesRegex(ValueError,'Instance limit'):build_index(a,9,8)

if __name__=='__main__':unittest.main()
