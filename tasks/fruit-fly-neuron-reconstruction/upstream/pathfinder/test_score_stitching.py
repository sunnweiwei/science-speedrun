import json
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
from score_stitching import run,export_pair
from common import sha256,write_json
from stitch_ffn import run as stitch
from test_stitch_ffn import fixture,xtemp_directory


@unittest.skipUnless(all(os.environ.get(k) for k in ('PATHFINDER_TEST_XTEMP','PATHFINDER_TEST_FFN_SOURCE','PATHFINDER_TEST_CONNECTOMICS_SOURCE','PATHFINDER_TEST_EVALUATOR')),'Xtemp, author sources and frozen evaluator required')
class PairedStitchTests(unittest.TestCase):
    def test_actual_stitch_and_score_use_identical_core_and_reference(self):
        with xtemp_directory() as folder:
            root=Path(folder);manifest=fixture(root);path=root/'manifest.json';write_json(path,manifest)
            stitched=root/'stitched'
            stitch(path,root,stitched,Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1)
            suite=root/'suite';suite.mkdir()
            # A cropped reference that straddles the x window boundary.
            metadata=dict(manifest['metadata'],origin_nm=[336]*3,shape=[6]*3)
            write_json(suite/'metadata.json',metadata)
            nodes=[dict(id=i,skeleton_id=1 if i<2 else 2,position_nm=[344,344 if i<2 else 408,344 if i%2==0 else 408]) for i in range(4)]
            write_json(suite/'skeletons.json',dict(axes='zyx',coordinate_units='nm',nodes=nodes,edges=[[0,1],[2,3]]))
            shutil.copyfile(os.environ['PATHFINDER_TEST_EVALUATOR'],suite/'run_length.py')
            write_json(suite/'suite.json',dict(schema='pathfinder-evaluation-suite/v1',boundary_margin_voxels=[0,0,0],files={name:sha256(suite/name) for name in ('metadata.json','skeletons.json','run_length.py')}))
            result=run(path,stitched,suite,root,root/'paired')
            self.assertTrue(result['complete'])
            self.assertGreater(result['results']['stitched']['nerl'],result['results']['unstitched']['nerl'])
            full=np.load(stitched/'segmentation.npy')
            after=np.load(root/'paired/stitched/segmentation.npy')
            before=np.load(root/'paired/unstitched/segmentation.npy')
            np.testing.assert_array_equal(after,full[1:7,1:7,1:7])
            np.testing.assert_array_equal(before>0,after>0)
            self.assertGreater(len(np.unique(before)),len(np.unique(after)))
            for name in ('stitched','unstitched'):
                report=json.loads((root/f'paired/score-{name}/metrics.json').read_text())
                self.assertEqual(report['sampled_nodes'],4)
                self.assertEqual(report['suite_sha256'],sha256(suite/'suite.json'))
            core=Path(json.loads((stitched/'receipt.json').read_text())['cores'][0]['path'])
            with core.open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'integrity'):export_pair(path,stitched,suite,root,root/'invalid')
            self.assertFalse((root/'invalid').exists())

if __name__=='__main__':unittest.main()
