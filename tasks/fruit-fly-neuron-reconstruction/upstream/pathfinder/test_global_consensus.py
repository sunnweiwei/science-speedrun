import os
from pathlib import Path
import unittest
import tempfile
import numpy as np
from global_consensus import intersect_inplace
from consensus_ffn import author_consensus


def assert_partition(test,a,b):
    np.testing.assert_array_equal(a==0,b==0)
    pairs=np.unique(np.stack([a.ravel(),b.ravel()],axis=1),axis=0)
    test.assertEqual(len(pairs),len(np.unique(pairs[:,0])))
    test.assertEqual(len(pairs),len(np.unique(pairs[:,1])))


class GlobalConsensusTests(unittest.TestCase):
    def test_memmap_round_trip_and_alias_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'a.npy';source=Path(folder)/'b.npy'
            a=np.arange(60,dtype=np.uint32).reshape(5,3,4)%7
            b=np.arange(60,dtype=np.uint32).reshape(5,3,4)%5
            expected=a.copy();intersect_inplace(expected,b,2,slab_depth=5,max_pairs=100)
            np.save(path,a);np.save(source,b)
            mapped=np.load(path,mmap_mode='r+');other=np.load(source,mmap_mode='r')
            try:
                with self.assertRaisesRegex(ValueError,'Independent'):intersect_inplace(mapped,mapped,0)
                intersect_inplace(mapped,other,2,slab_depth=2,max_pairs=100)
                mapped.flush()
            finally:mapped._mmap.close();other._mmap.close()
            np.testing.assert_array_equal(np.load(path),expected)
            np.testing.assert_array_equal(np.load(source),b)

    def test_counts_cross_slabs_and_background_is_asymmetric(self):
        a=np.array([1,2,1,2,1,0,1,0],np.uint32).reshape(4,1,2)
        b=np.array([0,5,0,5,0,8,0,8],np.uint32).reshape(a.shape)
        intersect_inplace(a,b,3,slab_depth=1,max_pairs=10)
        np.testing.assert_array_equal(a.ravel(),[1,0,1,0,1,0,1,0])

    def test_resource_failure_does_not_modify_input(self):
        a=np.arange(24,dtype=np.uint32).reshape(4,2,3);before=a.copy();b=np.ones_like(a)
        with self.assertRaisesRegex(ValueError,'pair table'):intersect_inplace(a,b,0,max_pairs=3,slab_depth=1)
        np.testing.assert_array_equal(a,before)
        with self.assertRaisesRegex(ValueError,'memory'):intersect_inplace(a,b,0,memory_gib=.00001)
        np.testing.assert_array_equal(a,before)

    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_FFN_SOURCE') and os.environ.get('PATHFINDER_TEST_CONNECTOMICS_SOURCE'),'Pinned author repositories required')
    def test_global_author_oracle_with_cross_slab_ids_and_ordered_runs(self):
        function,_=author_consensus(Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']))
        for seed in range(30):
            rng=np.random.default_rng(seed);inputs=[rng.integers(0,5,(7,6,5),dtype=np.uint32) for _ in range(2+seed%5)]
            # Exercise large uint32 IDs without changing their partition.
            for a in inputs:a[a>0]+=np.uint32(2**32-8)
            for minimum in (0,1,3,20):
                expected=inputs[0].astype(np.uint64);actual=inputs[0].copy()
                for b in inputs[1:]:
                    function(expected,b.astype(np.uint64),minimum)
                    intersect_inplace(actual,b,minimum,slab_depth=1+seed%4,max_pairs=1000)
                    assert_partition(self,actual,expected)

if __name__=='__main__':unittest.main()
