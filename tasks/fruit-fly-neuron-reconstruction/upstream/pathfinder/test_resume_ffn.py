import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dataset_contract import sha256
from resume_ffn import completed_tiles


class ResumeFFNTests(unittest.TestCase):
    def test_seed_orders_cannot_share_resume_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            path,expected,tiles=self.fixture(Path(folder))
            self.assertEqual(len(completed_tiles(path,dict(expected,seed_order='ascending'),tiles)[0]),1)
            with self.assertRaisesRegex(ValueError,'seed order'):
                completed_tiles(path,dict(expected,seed_order='descending'),tiles)
            receipt=json.loads(path.read_text());receipt['seed_order']='descending'
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'seed order'):completed_tiles(path,expected,tiles)
            self.assertEqual(len(completed_tiles(path,dict(expected,seed_order='descending'),tiles)[0]),1)

    def fixture(self, root):
        tiles=[dict(index=i,start=[4*i,0,0],stop=[4*i+4,4,4]) for i in range(3)]
        rows=[]
        for tile in tiles[:2]:
            path=root/f"{tile['index']}.npy"
            np.save(path,np.full((4,4,4),tile['index']*7+1,np.uint32))
            rows.append(dict(tile,path=str(path),sha256=sha256(path),budget_exhausted=tile['index']==1))
        expected=dict(id_stride=7,checkpoint_sha256='same-model',raw_sha256='same-image')
        receipt=root/'receipt.json'
        receipt.write_text(json.dumps(dict(expected,ground_truth_used=False,tile_results=rows)))
        return receipt,expected,tiles

    def test_reuses_only_complete_tiles_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path,expected,tiles=self.fixture(Path(folder))
            rows,provenance=completed_tiles(path,expected,tiles)
            self.assertEqual([r['index'] for r in rows],[0])
            self.assertEqual([t['index'] for t in tiles if t['index'] not in {r['index'] for r in rows}],[1,2])
            self.assertEqual(provenance['sha256'],sha256(path))
            self.assertEqual(rows[0]['sha256'],sha256(rows[0]['path']))

    def test_rejects_different_inputs_and_corrupted_completed_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path,expected,tiles=self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,'provenance'):
                completed_tiles(path,dict(expected,checkpoint_sha256='other-model'),tiles)
            with (Path(folder)/'0.npy').open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'bytes'):
                completed_tiles(path,expected,tiles)

    def test_rejects_foreign_ids_even_with_valid_file_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path,expected,tiles=self.fixture(Path(folder))
            arr=Path(folder)/'0.npy';np.save(arr,np.full((4,4,4),8,np.uint32))
            receipt=json.loads(path.read_text());receipt['tile_results'][0]['sha256']=sha256(arr)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'IDs'):
                completed_tiles(path,expected,tiles)

    def test_profile_comparison_excludes_partial_tiles(self):
        from profile_batched_ffn import compare_completed
        with tempfile.TemporaryDirectory() as folder:
            path,_,_=self.fixture(Path(folder));a=json.loads(path.read_text())
            b=json.loads(path.read_text())
            result=compare_completed(a,b)
            self.assertEqual(result['compared_complete_tiles'],1)
            self.assertTrue(result['exact_on_compared_tiles'])
            b['tile_results'][0]['budget_exhausted']=True
            self.assertFalse(compare_completed(a,b)['exact_on_compared_tiles'])

    def test_profile_reports_changed_voxels(self):
        from profile_batched_ffn import compare_completed
        with tempfile.TemporaryDirectory() as folder:
            path,_,_=self.fixture(Path(folder));a=json.loads(path.read_text());b=json.loads(path.read_text())
            arr=np.ones((4,4,4),np.uint32);arr[0,0,0]=0
            changed=Path(folder)/'changed.npy';np.save(changed,arr)
            b['tile_results'][0].update(path=str(changed),sha256=sha256(changed))
            result=compare_completed(a,b)
            self.assertEqual(result['changed_voxels'],1)
            self.assertFalse(result['exact_on_compared_tiles'])


if __name__=='__main__':unittest.main()
