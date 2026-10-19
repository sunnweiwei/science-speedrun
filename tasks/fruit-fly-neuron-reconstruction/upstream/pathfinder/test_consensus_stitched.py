import hashlib
import json
import os
from pathlib import Path
import unittest
import numpy as np
from consensus_stitched import run
from pack_stitched import run as pack
from stitch_ffn import run as stitch
from consensus_ffn import author_consensus
from test_stitch_ffn import fixture,xtemp_directory
from test_global_consensus import assert_partition
from common import sha256
from consensus_ffn import validate_checkpoint_runs,validate_six_runs
from prepare_prediction_context import parent_evidence
from unittest.mock import patch


class CheckpointProtocolTests(unittest.TestCase):
    def test_single_mode_rejects_mixed_weights_missing_duplicate_or_reversed_orders(self):
        rows=[dict(checkpoint_sha256='a',seed_order=order) for order in ('ascending','descending')]
        self.assertEqual(validate_checkpoint_runs(rows,1)['variants'],2)
        for bad in [rows[:1],rows+rows,[rows[0],rows[0]],list(reversed(rows)),
                    [rows[0],dict(checkpoint_sha256='b',seed_order='descending')]]:
            with self.assertRaises(ValueError):validate_checkpoint_runs(bad,1)
        with self.assertRaises(ValueError):validate_checkpoint_runs(rows)
        with self.assertRaises(ValueError):validate_six_runs(rows)

    def test_single_predictions_bind_to_exact_primary_asset(self):
        from consensus_stitched import bind_model_package
        from test_fs_support import TemporaryDirectory
        with TemporaryDirectory(dir=os.environ.get('PATHFINDER_TEST_XTEMP')) as folder:
            root=Path(folder);model=root/'model';model.mkdir();(model/'weights').mkdir()
            weight=model/'weights/ffn.pt';weight.write_bytes(b'pinned-single-model')
            manifest={'files':{'weights/ffn.pt':sha256(weight)}}
            (model/'package.json').write_text(json.dumps(manifest))
            check={'package_manifest_sha256':sha256(model/'package.json')}
            rows=[dict(checkpoint_sha256=sha256(weight),seed_order=order) for order in ('ascending','descending')]
            with patch('model_package.verify',return_value=check):
                binding,_=bind_model_package(model,rows,root,1)
                self.assertEqual(binding['checkpoints'],1)
                rows[1]['checkpoint_sha256']='unrelated'
                with self.assertRaisesRegex(ValueError,'Predictions differ'):bind_model_package(model,rows,root,1)


@unittest.skipUnless(all(os.environ.get(k) for k in ('PATHFINDER_TEST_XTEMP','PATHFINDER_TEST_FFN_SOURCE','PATHFINDER_TEST_CONNECTOMICS_SOURCE')),'Xtemp and author sources required')
class StitchedConsensusTests(unittest.TestCase):
    def test_six_synthetic_stitched_packages_match_author_global_oracle(self):
        with xtemp_directory() as folder:
            root=Path(folder);packages=[];arrays=[]
            ffn=Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']);connect=Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE'])
            oracle,_=author_consensus(ffn,connect)
            for i in range(6):
                variant=root/f'variant-{i}';variant.mkdir();manifest=fixture(variant)
                checkpoint=hashlib.sha256(f'synthetic-checkpoint-{i//2}'.encode()).hexdigest();order='descending' if i%2 else 'ascending'
                manifest.update(checkpoint_sha256=checkpoint,seed_order=order)
                for row in manifest['windows']:
                    row.update(checkpoint_sha256=checkpoint,seed_order=order)
                    if i:
                        a=np.load(row['path']);z=np.indices(a.shape)[0]+row['start'][0];a[z==21+i%4]=0
                        np.save(row['path'],a);row['sha256']=sha256(row['path'])
                path=variant/'manifest.json';path.write_text(json.dumps(manifest));prediction=variant/'stitched'
                stitch(path,root,prediction,ffn,connect,1)
                arrays.append(np.load(prediction/'segmentation.npy'))
                package=variant/'package';pack(prediction,path,root,package,100000,2);packages.append(str(package))
            # Original full NPY can be archived: the verified package is sufficient.
            (root/'variant-0/stitched/segmentation.npy').unlink()
            spec=root/'spec.json';spec.write_text(json.dumps(packages))
            for minimum in (0,3):
                output=root/f'consensus-{minimum}';result=run(spec,root,output,minimum,2,2,1000)
                self.assertTrue(result['complete']);self.assertEqual(result['required_six_runs']['variants'],6)
                expected=arrays[0].astype(np.uint64)
                for b in arrays[1:]:oracle(expected,b.astype(np.uint64),minimum)
                assert_partition(self,np.load(output/'segmentation.npy'),expected)
                self.assertEqual(len(result['stages']),5)
            spec.write_text(json.dumps(packages[:2]))
            single=root/'single';result=run(spec,root,single,0,2,2,1000,checkpoint_count=1)
            expected=arrays[0].astype(np.uint64);oracle(expected,arrays[1].astype(np.uint64),0)
            assert_partition(self,np.load(single/'segmentation.npy'),expected)
            self.assertEqual(result['schema'],'stitched-single-checkpoint-consensus/v1')
            self.assertNotIn('required_six_runs',result)
            self.assertEqual(result['required_checkpoint_runs']['variants'],2)
            self.assertEqual(len(result['stages']),1)
            self.assertEqual(parent_evidence(single,root)[0]['checkpoint_count'],1)
            with self.assertRaises(ValueError):run(spec,root,root/'implicit-single',0,2,2,1000)
            spec.write_text(json.dumps(packages))
            with self.assertRaisesRegex(ValueError,'memory'):run(spec,root,root/'small-memory',0,.01,2,1000)
            self.assertFalse((root/'small-memory').exists())
            with self.assertRaisesRegex(ValueError,'pair table'):run(spec,root,root/'small-table',0,2,2,1)
            self.assertFalse((root/'small-table/receipt.json').exists())
            spec.write_text(json.dumps(packages[:-1]+[packages[0]]))
            with self.assertRaisesRegex(ValueError,'checkpoint'):run(spec,root,root/'duplicate',0,2,2,1000)
            self.assertFalse((root/'duplicate').exists())
            spec.write_text(json.dumps(packages))
            archive=Path(packages[-1])/'segmentation.npz'
            with archive.open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):run(spec,root,root/'corrupt',0,2,2,1000)
            self.assertFalse((root/'corrupt').exists())

if __name__=='__main__':unittest.main()
