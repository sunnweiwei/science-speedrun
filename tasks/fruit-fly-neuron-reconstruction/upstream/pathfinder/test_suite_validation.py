import json,os,unittest
from pathlib import Path
from suite_validation import suite_pins
from freeze_evaluation import freeze
from common import sha256,write_json
from test_fs_support import TemporaryDirectory


@unittest.skipUnless(os.environ.get('PATHFINDER_TEST_EVALUATOR') and os.environ.get('PATHFINDER_TEST_XTEMP'),'Pinned author evaluator and Xtemp required')
class FrozenSuiteTests(unittest.TestCase):
    def fixture(self,root):
        meta=dict(axes='zyx',coordinate_convention='voxel_outer_corner',shape=[8]*3,origin_nm=[0]*3,voxel_size_nm=[16]*3)
        write_json(root/'metadata.json',meta)
        write_json(root/'skeletons.json',dict(axes='zyx',coordinate_units='nm',nodes=[dict(id=i,skeleton_id=1,position_nm=[16*i+8,8,8]) for i in range(4)],edges=[[0,1],[1,2],[2,3]]))
        freeze(root/'skeletons.json',root/'metadata.json',Path(os.environ['PATHFINDER_TEST_EVALUATOR']),root/'suite')
        return meta

    def test_frozen_suite_rejects_grid_and_byte_changes(self):
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            root=Path(folder);meta=self.fixture(root);pins=suite_pins(root/'suite',meta);self.assertEqual(len(pins),4)
            with self.assertRaisesRegex(ValueError,'geometry'):suite_pins(root/'suite',dict(meta,origin_nm=[16]*3))
            (root/'suite/skeletons.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'input changed'):suite_pins(root/'suite',meta)

    def test_frozen_core_inside_larger_prediction_does_not_require_equal_extents(self):
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            root=Path(folder);meta=self.fixture(root)
            enclosing=dict(meta,shape=[12]*3,origin_nm=[-32]*3)
            before=sha256(root/'suite/suite.json')
            self.assertEqual(len(suite_pins(root/'suite',enclosing)),4)
            self.assertEqual(before,sha256(root/'suite/suite.json'))
            with self.assertRaisesRegex(ValueError,'aligned'):suite_pins(root/'suite',dict(enclosing,origin_nm=[-31]*3))
            with self.assertRaisesRegex(ValueError,'spacing'):suite_pins(root/'suite',dict(enclosing,voxel_size_nm=[8]*3))

    def test_historical_report_text_preserves_frozen_suite_but_scoring_edit_fails(self):
        from suite_validation import CODE,verifier_matches
        historical='7c8cebd237caf8218dff7ee7aed049cd7e91c8675e34edc7a2a78babc25963d7'
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            root=Path(folder);meta=self.fixture(root)
            manifest=json.loads((root/'suite/suite.json').read_text())
            manifest['verifier_code']['verify_prediction.py']=historical
            write_json(root/'suite/suite.json',manifest)
            before=sha256(root/'suite/suite.json')
            suite_pins(root/'suite',meta)
            self.assertEqual(before,sha256(root/'suite/suite.json'))
            target=root/'verify_prediction.py'
            original=(CODE/'verify_prediction.py').read_text()
            self.assertIn("reward=",original)
            target.write_text(original.replace('reward=', 'reward=1+',1))
            self.assertFalse(verifier_matches(target,historical))
            self.assertFalse(verifier_matches(CODE/'verify_prediction.py','0'*64))

    def test_current_scoring_retains_controls_when_shape_is_worse(self):
        # Real frozen scorer, synthetic predictions; no GPU execution claim.
        import numpy as np
        from verify_single import score
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as folder:
            root=Path(folder);meta=self.fixture(root);pred=root/'prediction'
            for name in ('context','stitched','submission'):
                (pred/name).mkdir(parents=True)
                write_json(pred/name/'metadata.json',meta)
            base=np.ones((8,8,8),np.uint32);base[2:]=2
            np.save(pred/'stitched/segmentation.npy',base)
            np.save(pred/'submission/segmentation.npy',np.zeros_like(base))
            write_json(pred/'baseline.json',dict(base_sha256=sha256(pred/'stitched/segmentation.npy'),
                mapping={'2':1},complete=True,missing_pairs=0,incomplete_pairs=0,accepted_pairs=1))
            write_json(pred/'execution.json',dict(complete=True,protocol='single-checkpoint-fast/v1',
                prediction_sha256=sha256(pred/'submission/segmentation.npy')))
            logs=root/'logs';logs.mkdir()
            score(root,root/'suite',logs)
            result=json.loads((logs/'reward.json').read_text())
            self.assertEqual(result['reward'],0)
            self.assertAlmostEqual(result['methods']['shared-proposal-ffn']['nerl'],1)
            self.assertGreater(result['methods']['ffn']['nerl'],0)
            self.assertLess(result['methods']['ffn']['nerl'],1)


class CommonPinTests(unittest.TestCase):
    def test_current_common_is_the_one_accepted_renamed_file(self):
        from suite_validation import COMMON_RENAMED,CODE
        self.assertEqual(list(COMMON_RENAMED.values()),[sha256(CODE.parent/'common.py')])

if __name__=='__main__':unittest.main()
