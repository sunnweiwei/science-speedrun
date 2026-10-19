import copy
import json
from pathlib import Path
import unittest
from compare_report import compare


class PairedReferenceTests(unittest.TestCase):
    def test_local_gain_is_not_a_paper_gap(self):
        ref=json.loads(Path(__file__).with_name('paper_reference.json').read_text())
        before=dict(nerl=.3,erl_nm=30,split_errors=8,merge_errors=1,omitted_node_fraction=.1,
            maximum_erl_nm=100,total_path_length_nm=500,evaluator_sha256='eval',protocol='local',
            boundary_margin_voxels=[0]*3,inputs=dict(skeletons='gt',metadata='roi',segmentation='old'))
        after=copy.deepcopy(before);after['nerl']=.4;after['inputs']['segmentation']='new'
        r=compare(ref,after,before)
        self.assertAlmostEqual(r['local_change']['metrics']['nerl']['delta'],.1)
        self.assertFalse(r['directly_comparable']);self.assertIsNone(r['numeric_gap'])
        self.assertTrue(compare(ref,before,before)['local_change']['identical_prediction_bytes'])
        after['core_evaluator_adapter_sha256']='different-core'
        with self.assertRaises(ValueError):compare(ref,after,before)
        del after['core_evaluator_adapter_sha256']
        after['inputs']['skeletons']='other'
        with self.assertRaises(ValueError):compare(ref,after,before)

if __name__=='__main__':unittest.main()
