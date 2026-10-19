import unittest
import numpy as np
from merge_ffn_agglomeration import union_mapping,remap,combine,combine_ensemble


class GlobalMergeTest(unittest.TestCase):
    def ensemble_fixture(self):
        plan=dict(base_sha256='b',decisions=[dict(pair=[1,2])])
        shards=[]
        for i in range(3):
            row=dict(pair=[1,2],complete=True,accepted=True,iou=.95,
                     consistency=[[.95]*2]*2,prediction_voxels=[100,100],deleted_voxels=[0,0])
            shards.append(dict(shards=1,shard_index=0,checkpoint_sha256=str(i),raw_sha256='r',
                               base_sha256='b',decisions_sha256='d',results=[row],ground_truth_used=False))
        return plan,shards

    def test_ensemble_is_three_way_conjunction_not_majority_vote(self):
        plan,shards=self.ensemble_fixture()
        self.assertEqual(combine_ensemble(plan,shards,'d')['mapping'],{'2':1})
        shards[2]['results'][0]['consistency']=[[.7]*2]*2
        self.assertEqual(combine_ensemble(plan,shards,'d')['accepted_pairs'],0)
        with self.assertRaises(ValueError):combine_ensemble(plan,shards[:2],'d')
        shards[2]['results']=[]
        with self.assertRaises(ValueError):combine_ensemble(plan,shards,'d')

    def test_ensemble_rejects_unrelated_or_incomplete_model_asset_set(self):
        plan,shards=self.ensemble_fixture()
        expected=['2','0','1']
        self.assertEqual(combine_ensemble(plan,shards,'d',expected_checkpoints=expected)['mapping'],{'2':1})
        for bad in (['0','1','other'],['0','1'],['0','1','1']):
            with self.assertRaisesRegex(ValueError,'verified model package'):
                combine_ensemble(plan,shards,'d',expected_checkpoints=bad)

    def test_paper_reduction_reuses_manual_scores_without_trusting_acceptance(self):
        plan,shards=self.ensemble_fixture()
        row=shards[0]['results'][0];row['deleted_voxels']=[100,100];row['accepted']=False
        self.assertEqual(combine(plan,shards[:1],'d')['accepted_pairs'],0)
        self.assertEqual(combine(plan,shards[:1],'d',rule='pathfinder2025')['accepted_pairs'],1)
        self.assertEqual(combine_ensemble(plan,shards,'d')['accepted_pairs'],1)
        row['accepted']=True
        with self.assertRaisesRegex(ValueError,'Stored decision'):combine_ensemble(plan,shards,'d')

    def test_disk_table_same_scores_and_coverage_rejections(self):
        import copy,os
        from pathlib import Path
        from test_fs_support import TemporaryDirectory
        from decision_table import write_table,load_rows
        plan,shards=self.ensemble_fixture()
        plan['decisions']=[dict(pair=[1,2]),dict(pair=[2,3]),dict(pair=[3,4])]
        for shard in shards:
            template=shard['results'][0]
            shard['results']=[dict(template,pair=d['pair']) for d in plan['decisions']]
        expected=combine_ensemble(plan,shards,'d')
        with TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']) as tmp:
            descriptor=write_table(Path(tmp)/'table.npy',
                ((tuple(d['pair']),0.,(0,0,0)) for d in plan['decisions']),3,[0,0,0])
            disk_plan=dict(base_sha256='b',decision_table=descriptor)
            rows=load_rows(disk_plan,Path(tmp)/'manifest.json')
            disk_plan['decisions']=rows
            try:
                self.assertEqual(combine_ensemble(disk_plan,shards,'d'),expected)
                for pair in ([1,2],[2,3],[3,4]):self.assertEqual(rows[rows.pair_index(pair)]['pair'],pair)
                for pair in ([1,3],[4,5],[0,1]):
                    with self.assertRaises(ValueError):rows.pair_index(pair)
                duplicate=copy.deepcopy(shards);duplicate[0]['results'].append(duplicate[0]['results'][0])
                with self.assertRaisesRegex(ValueError,'duplicate'):combine_ensemble(disk_plan,duplicate,'d')
                missing=copy.deepcopy(shards);missing[0]['results'].pop()
                with self.assertRaisesRegex(ValueError,'incomplete'):combine_ensemble(disk_plan,missing,'d')
                # Correct global identities in the wrong round-robin shard fail.
                wrong=[dict(shards[0],shards=2,shard_index=0,results=shards[0]['results']),
                       dict(shards[0],shards=2,shard_index=1,results=[])]
                with self.assertRaisesRegex(ValueError,'Unexpected'):combine(disk_plan,wrong,'d')
            finally:rows.array._mmap.close()

    def test_global_transitivity_before_cropping(self):
        # Intermediate ID2 is outside the scored crop but connects1 and3.
        mapping=union_mapping([[1,2],[2,3]])
        np.testing.assert_array_equal(remap(np.array([[[0,1,3,99]]],np.uint64),mapping),[[[0,1,1,99]]])

    def test_incomplete_requires_explicit_mode(self):
        plan=dict(base_sha256='b',decisions=[dict(pair=[1,2])])
        shard=dict(shards=1,shard_index=0,checkpoint_sha256='c',raw_sha256='r',base_sha256='b',
                   decisions_sha256='d',results=[],ground_truth_used=False)
        with self.assertRaises(ValueError):combine(plan,[shard],'d')
        result=combine(plan,[shard],'d',allow_partial=True)
        self.assertFalse(result['complete']);self.assertEqual(result['missing_pairs'],1)

    def test_shard_source_mismatch_rejected(self):
        plan=dict(base_sha256='b',decisions=[])
        shard=dict(shards=1,shard_index=0,checkpoint_sha256='c',raw_sha256='r',base_sha256='wrong',
                   decisions_sha256='d',results=[],ground_truth_used=False)
        with self.assertRaises(ValueError):combine(plan,[shard],'d')


if __name__=='__main__':unittest.main()
