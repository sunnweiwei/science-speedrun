import copy
import unittest
from paired_regions import summarize


class RegionalComparisonTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(regions=[dict(region_id='a'),dict(region_id='b')],nested_scale_voxels=[256],
            training_exposure='unknown',statistics='descriptive')
        self.check=dict(investigation_required=False,nerl=.596)
        self.metric=dict(nerl=.1,erl_nm=10,split_errors=4,merge_errors=2,omitted_node_fraction=.2,
            maximum_erl_nm=100,total_path_length_nm=500,evaluator_sha256='eval',protocol='physical',
            boundary_margin_voxels=[0]*3,adapter_sha256='adapter',core_evaluator_adapter_sha256='core',
            topology_sha256='topology',suite_sha256='suite',inputs=dict(skeletons='gt',metadata='roi',segmentation='labels'))
        self.row=dict(region_id='a',core_voxels=256,ffn=copy.deepcopy(self.metric),full=copy.deepcopy(self.metric),baseline=copy.deepcopy(self.metric))

    def test_small_gain_preserved_without_claiming_sota_or_complete(self):
        self.row['full'].update(nerl=.1000216,erl_nm=10.00216)
        out=summarize(self.plan,[self.row],self.check)
        self.assertAlmostEqual(out['aggregates'][0]['mean_nerl_full_minus_baseline'],.0000216)
        self.assertFalse(out['all_expected_regions_scored'])
        self.assertIsNone(out['direct_pathfinder_gap'])
        self.assertIsNone(out['aggregates'][0]['confidence_interval'])

    def test_incompatible_or_unverified_results_fail(self):
        with self.assertRaises(ValueError):summarize(self.plan,[self.row],dict(self.check,investigation_required=True))
        with self.assertRaises(ValueError):summarize(self.plan,[self.row,self.row],self.check)
        for nerl in (True,float('nan'),.1):
            with self.assertRaises(ValueError):summarize(self.plan,[self.row],dict(self.check,nerl=nerl))
        duplicate=copy.deepcopy(self.row);duplicate['region_id']='b'
        with self.assertRaises(ValueError):summarize(self.plan,[self.row,duplicate],self.check)
        for change in (dict(suite_sha256='other'),dict(topology_sha256='other'),dict(adapter_sha256=None)):
            row=copy.deepcopy(self.row);row['baseline'].update(change)
            with self.assertRaises(ValueError):summarize(self.plan,[row],self.check)

    def test_invalid_region_retained_and_not_claimed_complete(self):
        row=dict(region_id='b',core_voxels=256,status='invalid_reference',reason='No reference edges')
        out=summarize(self.plan,[self.row,row],self.check)
        self.assertEqual(out['missing'],[])
        self.assertEqual(out['invalid_reference_count'],1)
        self.assertFalse(out['all_expected_regions_scored'])


if __name__=='__main__':unittest.main()
