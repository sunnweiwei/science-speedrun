import os
from pathlib import Path
from test_fs_support import TemporaryDirectory
import unittest
import numpy as np
from decision_centroid import reduce_centroid
from decision_spill import partition_votes,reduce_partitions,sorted_results

class SpillTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP'))
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'spill'

    def test_exact_votes_centroids_ties_across_partitions(self):
        rng=np.random.default_rng(73);rows=[]
        for _ in range(200):
            a=int(rng.integers(1,12));b=int(rng.integers(12,30))
            rows.append(((a,b),float(rng.integers(0,3)),rng.integers(0,20,(int(rng.integers(1,8)),3)),int(rng.integers(0,7))))
        rows += [((100,101),0.,np.array([[0,0,0],[0,0,0],[0,0,2]]),1),
                 ((100,101),0.,np.array([[0,0,10]]),0)]
        expected=reduce_centroid(lambda:iter(rows),max_pairs=500)
        manifest=partition_votes(lambda:iter(rows),self.path,partitions=7,max_disk_bytes=10**7)
        paths,count=reduce_partitions(self.path,manifest['counts'],max_pairs_per_partition=500,max_pairs=500,max_disk_bytes=10**7)
        self.assertEqual(count,len(expected));self.assertEqual(list(sorted_results(paths)),expected)

    def test_parallel_reduction_matches_serial_with_locked_global_bounds(self):
        from decision_spill import reduce_partitions_parallel
        rng=np.random.default_rng(9)
        rows=[((i,i+100),float(i%3),rng.integers(0,20,(5,3)),i%7) for i in range(1,100)]
        rows+=rows[:20]
        expected=reduce_centroid(lambda:iter(rows),max_pairs=100)
        manifest=partition_votes(lambda:iter(rows),self.path,partitions=7,max_disk_bytes=10**7)
        paths,count=reduce_partitions_parallel(self.path,manifest['counts'],workers=2,
            max_pairs_per_partition=100,max_pairs=100,max_disk_bytes=10**7)
        self.assertEqual(count,len(expected));self.assertEqual(list(sorted_results(paths)),expected)

    def test_parallel_global_pair_limit_rejects_partial_collection(self):
        from decision_spill import reduce_partitions_parallel
        rows=[((i,i+100),0.,np.zeros((1,3),int),0) for i in range(1,20)]
        manifest=partition_votes(lambda:iter(rows),self.path,partitions=7,max_disk_bytes=10**7)
        with self.assertRaisesRegex(ValueError,'pair limit'):
            reduce_partitions_parallel(self.path,manifest['counts'],workers=2,
                max_pairs_per_partition=100,max_pairs=1,max_disk_bytes=10**7)

    def test_disk_bound_preserves_failure(self):
        rows=[((1,2),0.,np.zeros((3,3),dtype=int),0)]
        with self.assertRaisesRegex(ValueError,'disk limit'):
            partition_votes(lambda:iter(rows),self.path,partitions=2,max_disk_bytes=1)
        self.assertFalse(list(self.path.glob('result-*')))

    def test_pair_bound_never_publishes_complete_output(self):
        rows=[((1,2),0.,np.zeros((1,3),dtype=int),0),((2,3),0.,np.ones((1,3),dtype=int),1)]
        m=partition_votes(lambda:iter(rows),self.path,partitions=2,max_disk_bytes=10000)
        with self.assertRaisesRegex(ValueError,'pair limit'):
            reduce_partitions(self.path,m['counts'],max_pairs_per_partition=10,max_pairs=1,max_disk_bytes=10000)

if __name__=='__main__':unittest.main()
