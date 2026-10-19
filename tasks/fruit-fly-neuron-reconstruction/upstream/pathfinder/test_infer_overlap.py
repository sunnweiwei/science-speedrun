import copy
from pathlib import Path
from test_fs_support import TemporaryDirectory
import unittest
from infer_overlap import merge,select_windows
from test_stitch_ffn import fixture


def shard_fixture(folder):
    manifest=fixture(Path(folder));manifest['source_sha256']={'source.py':'frozen'}
    return [dict(manifest,schema='overlapping-ffn-shard/v1',complete=True,shards=4,shard_index=i,
                 windows=[r for r in manifest['windows'] if r['index']%4==i]) for i in range(4)]


class MergeOverlapTests(unittest.TestCase):
    def test_backend_provenance_cannot_mix_and_legacy_defaults_to_eager(self):
        with TemporaryDirectory() as folder:
            shards=shard_fixture(folder)
            self.assertEqual(merge(shards)['inference_backend'],'eager')
            shards[0]['inference_backend']='norm32-graph'
            with self.assertRaisesRegex(ValueError,'backend'):merge(shards)
            for backend in ('norm32','norm32-graph'):
                for shard in shards:shard['inference_backend']=backend
                self.assertEqual(merge(shards)['inference_backend'],backend)

    def test_recovery_selection_preserves_geometry_and_rejects_false_full_coverage(self):
        rows=[dict(index=i,start=[i,1,2],stop=[i+3,4,5]) for i in range(3)]
        self.assertEqual(select_windows(rows,[2,0]),[rows[0],rows[2]])
        for indices in ([],[0,0],[4],[True],None):
            with self.assertRaises(ValueError):select_windows(rows,indices)
        with self.assertRaisesRegex(ValueError,'recovery join'):
            merge([dict(window_selection={'indices':[0]})])

    def test_balanced_global_shards_require_exact_ownership(self):
        from window_assignment import assign
        with TemporaryDirectory() as folder:
            shards=shard_fixture(folder);rows=[r for s in shards for r in s['windows']]
            owners=assign(rows,4,'voxel-balanced-v1')
            for i,s in enumerate(shards):
                s['assignment_policy']='voxel-balanced-v1'
                s['windows']=[r for r in rows if owners[r['index']]==i]
            self.assertEqual(len(merge(shards)['windows']),8)
            bad=copy.deepcopy(shards);bad[1]['assignment_policy']='modulo'
            with self.assertRaisesRegex(ValueError,'assignment'):merge(bad)
            bad=copy.deepcopy(shards);bad[0]['windows'].pop()
            with self.assertRaisesRegex(ValueError,'assignment'):merge(bad)

    def test_complete_four_shards_preserve_all_windows(self):
        with TemporaryDirectory() as folder:
            shards=shard_fixture(folder);result=merge(list(reversed(shards)))
            self.assertEqual(sorted(r['index'] for r in result['windows']),list(range(8)))
            self.assertEqual(result['schema'],'overlapping-ffn-windows/v1')

    def test_wrong_worker_assignment_and_source_rejected(self):
        with TemporaryDirectory() as folder:
            shards=shard_fixture(folder)
            bad=copy.deepcopy(shards)
            bad[0]['windows'][0],bad[1]['windows'][0]=bad[1]['windows'][0],bad[0]['windows'][0]
            with self.assertRaisesRegex(ValueError,'assignment'):merge(bad)
            bad=copy.deepcopy(shards);bad[1]['source_sha256']={'source.py':'changed'}
            with self.assertRaisesRegex(ValueError,'provenance'):merge(bad)
            bad=copy.deepcopy(shards);bad[1]['boundary_image_shape']=[64]*3
            with self.assertRaisesRegex(ValueError,'boundary'):merge(bad)
            bad=copy.deepcopy(shards);bad[1]['compressed_window_bytes']=1000
            with self.assertRaisesRegex(ValueError,'storage'):merge(bad)
            bad=copy.deepcopy(shards);bad[1]['compression_level']=6
            with self.assertRaisesRegex(ValueError,'compression level'):merge(bad)

    def test_partial_and_duplicate_shards_rejected(self):
        with TemporaryDirectory() as folder:
            shards=shard_fixture(folder)
            bad=copy.deepcopy(shards);bad[1]['complete']=False
            with self.assertRaisesRegex(ValueError,'Complete'):merge(bad)
            with self.assertRaisesRegex(ValueError,'unique'):merge(shards[:-1]+[shards[0]])
            with self.assertRaisesRegex(ValueError,'unique'):merge(shards[:-1])


if __name__=='__main__':unittest.main()
