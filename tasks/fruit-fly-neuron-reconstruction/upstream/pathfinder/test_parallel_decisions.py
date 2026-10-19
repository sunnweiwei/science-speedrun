import os,unittest
from pathlib import Path
import numpy as np
from test_fs_support import TemporaryDirectory
from decision_centroid import groups,reduce_centroid
from decision_spill import sorted_results
from parallel_decisions import extract,reduce

class ParallelExtractionTest(unittest.TestCase):
    def test_cross_chunk_faces_votes_and_global_ties_are_exact(self):
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
            tmp=Path(tmp);rng=np.random.default_rng(913)
            expanded=rng.integers(0,12,(13,15,17),dtype=np.uint32)
            distance=rng.integers(0,4,expanded.shape).astype(np.float32)
            np.save(tmp/'expanded.npy',expanded);np.save(tmp/'distance.npy',distance)
            roi=[1,12,2,14,1,16]
            expected=reduce_centroid(lambda:groups(expanded,distance,roi,tile=5),max_pairs=1000)
            m=extract(tmp/'expanded.npy',tmp/'distance.npy',roi,tmp/'spill',workers=2,partitions=7,tile=5,max_disk_bytes=10**7)
            paths,count=reduce(tmp/'spill',m,workers=2,max_pairs_per_partition=1000,max_pairs=1000,max_disk_bytes=10**7)
            self.assertEqual(count,len(expected));self.assertEqual(list(sorted_results(paths)),expected)

if __name__=='__main__':unittest.main()
