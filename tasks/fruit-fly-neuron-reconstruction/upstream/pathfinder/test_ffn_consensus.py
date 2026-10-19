import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from consensus_ffn import verify_input, author_consensus, consensus_blocks, validate_six_runs
from common import sha256


def fixture(path, data=None):
    path.mkdir()
    if data is None:
        data = np.ones((4,4,4), dtype=np.uint32)
    np.save(path/'tile.npy', data)
    np.save(path/'segmentation.npy', data)
    receipt = dict(shards=1, shard_index=0, all_shard_tiles_complete=True,
                   ground_truth_used=False, roi=[16,20]*3, tile_size=4, id_stride=7,
                   checkpoint_sha256='synthetic-model', raw_sha256='synthetic-image',
                   metadata=dict(axes='zyx', origin_nm=[256]*3, voxel_size_nm=[16]*3,
                                 shape=[4]*3, coordinate_convention='voxel_outer_corner'),
                   tile_results=[dict(index=0,start=[16]*3,stop=[20]*3,
                                      path=str((path/'tile.npy').resolve()),
                                      sha256=sha256(path/'tile.npy'),budget_exhausted=False)])
    (path/'receipt.json').write_text(json.dumps(receipt))
    return receipt


class ConsensusInputTests(unittest.TestCase):
    def test_six_run_variant_coverage(self):
        rows=[dict(checkpoint_sha256=c,seed_order=o) for c in ('a','b','c')
              for o in ('ascending','descending')]
        self.assertEqual(validate_six_runs(rows)['variants'],6)
        self.assertEqual(validate_six_runs(list(reversed(rows)))['variants'],6)
        for bad in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]],
                    [dict(r,seed_order='ascending') for r in rows],
                    [dict(r,checkpoint_sha256='a') for r in rows]):
            with self.assertRaisesRegex(ValueError,'three checkpoint'):
                validate_six_runs(bad)


    @unittest.skipUnless(os.environ.get('PATHFINDER_TEST_FFN_SOURCE') and os.environ.get('PATHFINDER_TEST_CONNECTOMICS_SOURCE'),
                         'Pinned author checkouts required for actual global-function comparison')
    def test_tiled_partition_matches_actual_global_author_function(self):
        official,_=author_consensus(Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),
                                   Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']))
        rng=np.random.default_rng(20261001)
        tiles=[dict(index=i,start=[i*4,0,0],stop=[i*4+4,5,6]) for i in range(3)]
        for case in range(150):
            first=rng.integers(0,6,(12,5,6),dtype=np.uint64)
            for i in range(3):
                block=first[i*4:i*4+4];block[block>0]+=i*10
            if case%3==0:first[first>0]+=2**40
            arrays=[first]+[rng.integers(0,8,first.shape,dtype=np.uint64) for _ in range(1+case%5)]
            if case%5==0:arrays[1].fill(0)
            minimum=case%11;whole=first.copy()
            for b in arrays[1:]:official(whole,b.copy(),minimum)
            tiled=np.zeros_like(first)
            for section,block,_ in consensus_blocks(arrays,tiles,np.array([0,0,0]),official,minimum):
                tiled[section]=block
            np.testing.assert_array_equal(whole==0,tiled==0)
            pairs=np.unique(np.stack((whole[whole>0],tiled[whole>0]),axis=1),axis=0)
            self.assertEqual(len(pairs),len(np.unique(whole[whole>0])))
            self.assertEqual(len(pairs),len(np.unique(tiled[tiled>0])))

    def test_hashed_assembly_survives_tile_archival_but_not_receipt_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source';r=fixture(source)
            p=Path(folder)/'assembled';p.mkdir()
            np.save(p/'segmentation.npy',np.ones((4,4,4),np.uint32))
            (p/'receipt.json').write_text(json.dumps(dict(all_tiles_complete=True,
                ground_truth_used=False,metadata=r['metadata'],
                segmentation_sha256=sha256(p/'segmentation.npy'),
                inputs=[dict(path=str(source/'receipt.json'),sha256=sha256(source/'receipt.json'))])))
            (source/'tile.npy').unlink()
            self.assertEqual(int(verify_input(p)[0].sum()),64)
            (source/'receipt.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt changed'):verify_input(p)

    def test_tile_memory_limit_checked_before_reading_tile_data(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'prediction';fixture(p)
            (p/'tile.npy').unlink()
            with self.assertRaisesRegex(ValueError,'memory limit'):verify_input(p,1)

    def test_verified_array_matches_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'prediction';fixture(p)
            array,_=verify_input(p)
            self.assertEqual(int(array.sum()),64)

    def test_complete_flag_cannot_hide_missing_or_partial_tile(self):
        for mode in ('missing','partial'):
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder)/'prediction';r=fixture(p)
                if mode=='missing':r['tile_results']=[]
                else:r['tile_results'][0]['budget_exhausted']=True
                (p/'receipt.json').write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError,'Incomplete'):verify_input(p)

    def test_changed_assembled_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'prediction';fixture(p)
            np.save(p/'segmentation.npy',np.zeros((4,4,4),np.uint32))
            with self.assertRaisesRegex(ValueError,'differs from verified'):verify_input(p)

    def test_foreign_tile_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'prediction';fixture(p,np.full((4,4,4),7,np.uint32))
            with self.assertRaisesRegex(ValueError,'namespace'):verify_input(p)


if __name__=='__main__':unittest.main()
