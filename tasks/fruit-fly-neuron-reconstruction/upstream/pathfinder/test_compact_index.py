import unittest
import numpy as np
from compact_index import build_table,records
from instance_index import build_index

class CompactIndexTests(unittest.TestCase):
    def test_exact_legacy_oracle_with_ties_holes_and_slab_boundaries(self):
        rng=np.random.default_rng(418)
        fixtures=[np.zeros((4,5,6),np.uint32),np.ones((4,5,6),np.uint32),rng.integers(0,23,(13,11,9),dtype=np.uint32)]
        a=np.zeros((9,8,7),np.uint32);a[0,0,0]=17;a[-1,-1,-1]=17;a[1:8,2,3]=1000;fixtures.append(a)
        for a in fixtures:
            oracle=build_index(a,2)
            for depth in (1,3,20):
                with self.subTest(shape=a.shape,depth=depth):
                    t=build_table(a,depth,max_label=1000,max_instances=1000)
                    self.assertEqual(list(records(t,a.shape)),oracle)
                    self.assertEqual(int(t['size'].sum()),np.count_nonzero(a))
    def test_bounds_are_failures_not_silent_filtering(self):
        a=np.arange(1,10,dtype=np.uint32).reshape(9,1,1)
        with self.assertRaisesRegex(ValueError,'label bound'):build_table(a,max_label=8,max_instances=9)
        with self.assertRaisesRegex(ValueError,'Instance bound'):build_table(a,max_label=9,max_instances=8)

if __name__=='__main__':unittest.main()
