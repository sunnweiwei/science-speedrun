import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt
from exact_fragment_edt import fragment_distance

class FragmentEDTTests(unittest.TestCase):
    def check(self,mask):np.testing.assert_array_equal(fragment_distance(mask),distance_transform_edt(mask))
    def test_sparse_dense_boundaries_and_all_foreground(self):
        shape=(19,21,23)
        self.check(np.zeros(shape,bool));self.check(np.ones(shape,bool))
        for point in [(0,0,0),(18,20,22),(0,10,22),(9,10,11)]:
            mask=np.zeros(shape,bool);mask[point]=True;self.check(mask)
        for axis in range(3):
            for side in (slice(0,6),slice(-6,None)):
                mask=np.zeros(shape,bool);sl=[slice(4,15)]*3;sl[axis]=side;mask[tuple(sl)]=True;self.check(mask)
    def test_disconnected_random_and_thin_axes(self):
        rng=np.random.default_rng(731)
        for shape in [(1,9,11),(13,1,15),(13,15,1),(17,19,21)]:
            for density in (.001,.01,.2,.9):self.check(rng.random(shape)<density)
        for _ in range(20):
            mask=np.zeros((31,33,35),bool)
            a=rng.integers(0,20,3);b=a+rng.integers(1,10,3)
            mask[tuple(slice(x,y) for x,y in zip(a,b))]=rng.random(tuple(b-a))>.2
            self.check(mask)

if __name__=='__main__':unittest.main()
