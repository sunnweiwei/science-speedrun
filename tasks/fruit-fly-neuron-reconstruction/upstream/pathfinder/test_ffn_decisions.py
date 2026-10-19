"""Compare the bounded adapter with actual pinned Google decision-point code."""
import os
from pathlib import Path
import unittest
import numpy as np
from prepare_ffn_decisions import author_function,generate


@unittest.skipUnless(os.environ.get('PATHFINDER_TEST_FFN_SOURCE') and
                     os.environ.get('PATHFINDER_TEST_CONNECTOMICS_SOURCE'),
                     'Pinned official repositories required')
class OfficialDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fn,box=author_function(Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),
                              Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']))
        cls.find=staticmethod(fn);cls.box=box

    def oracle(self,base,roi):
        lo=np.array(roi[::2]);hi=np.array(roi[1::2])
        found=self.find(base,[16]*3,max_distance=64,optimize_sparse=True,
                        subvol_box=self.box(start=lo[::-1],size=(hi-lo)[::-1]))
        return {tuple(map(int,pair)):(float(distance),tuple(map(int,np.array(xyz)[::-1]+lo)))
                for pair,(distance,xyz) in found.items()}

    def test_disk_partitions_match_official_random_ties(self):
        from connectomics.segmentation.labels import watershed_expand
        from decision_centroid import groups
        from decision_spill import partition_votes,reduce_partitions,sorted_results
        from test_fs_support import TemporaryDirectory
        for seed in range(3):
            base=np.zeros((120,120,120),np.uint32);rng=np.random.default_rng(seed)
            for label,point in enumerate(rng.integers(46,75,size=(24,3)),start=1):
                base[tuple(point)]=label*100003
            roi=[50,70]*3;expected=self.oracle(base,roi)
            expanded,distance=watershed_expand(base,[16]*3,max_distance=64)
            for tile in (8,13):
                with self.subTest(seed=seed,tile=tile), TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
                    folder=Path(tmp)/'spill'
                    manifest=partition_votes(lambda:groups(expanded,distance,roi,tile=tile),folder,
                                             partitions=7,max_disk_bytes=10**8)
                    paths,count=reduce_partitions(folder,manifest['counts'],max_pairs_per_partition=1000,
                                                   max_pairs=1000,max_disk_bytes=10**8)
                    actual={pair:(d,tuple(v+50 for v in point)) for pair,d,point in sorted_results(paths)}
                    self.assertEqual(count,len(expected));self.assertEqual(actual,expected)

    def test_production_cli_matches_whole_official_oracle(self):
        import json,subprocess,sys
        from test_fs_support import TemporaryDirectory
        base=np.zeros((120,124,128),np.uint32);roi=[50,70,53,73,56,76]
        base[55:59,58:62,61:65]=100001
        base[60:64,65:69,66:70]=900003
        expected=self.oracle(base,roi)
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            folder=Path(folder);np.save(folder/'base.npy',base)
            subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_ffn_decisions.py')),
                '--base',str(folder/'base.npy'),'--output',str(folder/'decisions.json'),
                '--ffn-source',os.environ['PATHFINDER_TEST_FFN_SOURCE'],
                '--connectomics-source',os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE'],
                '--roi',*map(str,roi),'--memory-gib','4','--max-pairs','100'],check=True,capture_output=True)
            result=json.loads((folder/'decisions.json').read_text())
        actual={tuple(r['pair']):(r['distance_nm'],tuple(r['center_zyx'])) for r in result['decisions']}
        self.assertEqual(actual,expected)
        self.assertIn('entire input',result['expansion_scope'])
        self.assertEqual(result['passes'],2)

    def test_parallel_disk_extraction_matches_official_global_watershed(self):
        from connectomics.segmentation.labels import watershed_expand
        from parallel_decisions import extract,reduce
        from decision_spill import sorted_results
        from test_fs_support import TemporaryDirectory
        base=np.zeros((120,120,120),np.uint32);rng=np.random.default_rng(17)
        for label,point in enumerate(rng.integers(46,75,size=(24,3)),start=1):base[tuple(point)]=label*100003
        roi=[50,70]*3;expected=self.oracle(base,roi)
        expanded,distance=watershed_expand(base,[16]*3,max_distance=64)
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
            tmp=Path(tmp);np.save(tmp/'expanded.npy',expanded);np.save(tmp/'distance.npy',distance)
            manifest=extract(tmp/'expanded.npy',tmp/'distance.npy',roi,tmp/'spill',workers=2,partitions=7,tile=8,max_disk_bytes=10**8)
            paths,count=reduce(tmp/'spill',manifest,workers=2,max_pairs_per_partition=1000,max_pairs=1000,max_disk_bytes=10**8)
            actual={pair:(d,tuple(x+50 for x in point)) for pair,d,point in sorted_results(paths)}
            self.assertEqual(count,len(expected));self.assertEqual(actual,expected)

    def test_disk_cli_matches_official_and_round_robin_shards(self):
        import json,subprocess,sys
        from test_fs_support import TemporaryDirectory
        from decision_table import load_rows
        base=np.zeros((120,120,120),np.uint32);roi=[50,70]*3
        rng=np.random.default_rng(4)
        for label,point in enumerate(rng.integers(46,75,size=(24,3)),start=1):
            base[tuple(point)]=label*100003
        expected=self.oracle(base,roi)
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
            folder=Path(tmp);np.save(folder/'base.npy',base);out=folder/'decisions.json'
            subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_ffn_decisions.py')),
                '--base',str(folder/'base.npy'),'--output',str(out),
                '--ffn-source',os.environ['PATHFINDER_TEST_FFN_SOURCE'],
                '--connectomics-source',os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE'],
                '--roi',*map(str,roi),'--memory-gib','4','--max-pairs','1000',
                '--spill-directory',str(folder/'spill'),'--spill-partitions','7','--spill-workers','2',
                '--spill-disk-gib','1','--max-pairs-per-partition','1000'],check=True,capture_output=True)
            plan=json.loads(out.read_text());rows=load_rows(plan,out)
            try:
                actual={tuple(r['pair']):(r['distance_nm'],tuple(r['center_zyx'])) for r in rows}
                self.assertEqual(actual,expected)
                legacy=list(rows)
                for rank in range(4):self.assertEqual(list(rows[rank::4]),legacy[rank::4])
                self.assertEqual(list(rows[::-1]),legacy[::-1])
                self.assertEqual(rows[-1],legacy[-1])
            finally:rows.array._mmap.close()
            with (folder/plan['decision_table']['file']).open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'identity mismatch'):load_rows(plan,out)

    def test_asymmetric_origin_matches_official_single_region_coordinates(self):
        base=np.zeros((120,124,128),np.uint64);roi=[50,70,53,73,56,76]
        base[55:59,58:62,61:65]=100001
        base[60:64,65:69,66:70]=900003
        expected=self.oracle(base,roi)
        self.assertEqual(set(expected),{(100001,900003)})
        actual=generate(base,roi,self.find,self.box,tile=64)
        observed={tuple(r['pair']):(r['distance_nm'],tuple(r['center_zyx'])) for r in actual['decisions']}
        self.assertEqual(observed,expected)

    def test_global_centroid_matches_official_with_unequal_chunk_votes(self):
        from connectomics.segmentation.labels import watershed_expand
        from decision_centroid import generate_global
        base=np.zeros((120,124,128),np.uint64);roi=[50,70,53,73,56,76]
        base[51:68,54:71,60:62]=17
        base[51:68,54:71,64:66]=900003
        expected=self.oracle(base,roi)
        self.assertEqual(set(expected),{(17,900003)})
        for tile in (8,13,64):
            with self.subTest(tile=tile):
                result=generate_global(base,roi,watershed_expand,memory_gib=4,tile=tile,max_pairs=10)
                actual={tuple(r['pair']):(r['distance_nm'],tuple(r['center_zyx'])) for r in result['decisions']}
                self.assertEqual(actual,expected)

    def test_global_centroid_matches_official_sparse_random_and_watershed_ties(self):
        from connectomics.segmentation.labels import watershed_expand
        from decision_centroid import generate_global
        for seed in range(4):
            base=np.zeros((120,120,120),np.uint64);rng=np.random.default_rng(seed)
            for label,point in enumerate(rng.integers(46,75,size=(24,3)),start=1):
                base[tuple(point)]=label*100003
            roi=[50,70]*3;expected=self.oracle(base,roi)
            self.assertTrue(expected)
            for tile in (8,13,64):
                with self.subTest(seed=seed,tile=tile):
                    result=generate_global(base,roi,watershed_expand,memory_gib=4,tile=tile,max_pairs=1000)
                    actual={tuple(r['pair']):(r['distance_nm'],tuple(r['center_zyx'])) for r in result['decisions']}
                    self.assertEqual(actual,expected)

    def test_chunk_faces_preserve_official_pairs_and_minimum_distances(self):
        # Adjacent fragments cross the first chunk boundary in each axis.
        # Global nearest-centroid tie selection is not claimed: the adapter
        # explicitly reduces equal-distance chunk proposals lexicographically.
        roi=[50,70]*3
        for axis in range(3):
            with self.subTest(axis=axis):
                base=np.zeros((120,120,120),np.uint64)
                a=[slice(61,63)]*3;b=list(a)
                a[axis]=slice(56,58);b[axis]=slice(58,60)
                base[tuple(a)]=17;base[tuple(b)]=900003
                expected=self.oracle(base,roi)
                self.assertEqual(set(expected),{(17,900003)})
                for tile in (8,13,64):
                    actual=generate(base,roi,self.find,self.box,tile=tile)
                    observed={tuple(r['pair']):r['distance_nm'] for r in actual['decisions']}
                    self.assertEqual(observed,{k:v[0] for k,v in expected.items()})
                    for row in actual['decisions']:
                        self.assertTrue(all(50<=v<70 for v in row['center_zyx']))


if __name__=='__main__':unittest.main()
