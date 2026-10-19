import unittest
import numpy as np
from decision_centroid import reduce_centroid


class GlobalReductionTests(unittest.TestCase):
    def test_retains_all_votes_instead_of_one_chunk_representative(self):
        # Chunk one has three votes; reducing it to one point would change the
        # centroid and select the distant point from chunk two.
        rows=[((1,2),0.,np.array([[0,0,0],[0,0,0],[0,0,2]]),1),
              ((1,2),0.,np.array([[0,0,10]]),0)]
        self.assertEqual(reduce_centroid(lambda:iter(rows),max_pairs=1),[((1,2),0.,(0,0,2))])

    def test_lower_minimum_replaces_prior_votes_and_offset_wins_tie(self):
        rows=[((1,2),3.,np.array([[0,0,100]]),0),
              ((1,2),1.,np.array([[0,0,0]]),5),
              ((1,2),1.,np.array([[0,0,2]]),1)]
        self.assertEqual(reduce_centroid(lambda:iter(rows),max_pairs=1),[((1,2),1.,(0,0,2))])
        self.assertEqual(reduce_centroid(lambda:iter(rows[::-1]),max_pairs=1),[((1,2),1.,(0,0,2))])

    def test_memory_rejection_happens_before_official_expansion(self):
        from decision_centroid import generate_global
        def forbidden(*args,**kwargs):raise AssertionError('No expansion allowed')
        base=np.zeros((120,120,120),np.uint32)
        with self.assertRaisesRegex(ValueError,'memory estimate'):
            generate_global(base,[50,70]*3,forbidden,memory_gib=.01)

    def test_pair_bound_fails_without_partial_output(self):
        rows=[((1,2),0.,np.zeros((1,3),int),0),((2,3),0.,np.zeros((1,3),int),0)]
        with self.assertRaisesRegex(ValueError,'pair limit'):
            reduce_centroid(lambda:iter(rows),max_pairs=1)

    def test_full_geometry_memory_preflight_without_materializing_volume(self):
        from decision_centroid import memory_preflight
        with self.assertRaisesRegex(ValueError,'memory estimate'):
            memory_preflight([1762]*3,[50,1712]*3,memory_gib=128)
        required=memory_preflight([1762]*3,[50,1712]*3,memory_gib=704)
        self.assertGreater(required,640*2**30)
        self.assertLess(required,704*2**30)


if __name__=='__main__':unittest.main()
