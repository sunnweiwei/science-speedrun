import copy
import unittest
from ffn_reconstruction_admission import admit_reconstruction


class ReconstructionGateTest(unittest.TestCase):
    def setUp(self):
        names=[f'variant-{i}' for i in range(6)]+['consensus']
        self.scores=dict(test_used=False,results=[dict(name=n,metrics=dict(
            nerl=.04 if n!='consensus' else .02,omitted_node_fraction=.3)) for n in names])
        self.receipts=[dict(all_shard_tiles_complete=True,ground_truth_used=False,
            tile_results=[{'budget_exhausted':False}],raw_sha256='raw',roi=[16,272]*3,
            metadata={'shape':[256]*3},tile_size=64,seed_order=['ascending','descending'][i%2],
            checkpoint_sha256=str(i//2),model_windows=100) for i in range(6)]
        self.candidate=copy.deepcopy(self.scores)
        self.candidate['results'][-1]['metrics']['nerl']=.025
        self.outputs=copy.deepcopy(self.receipts)
        self.outputs[2]['checkpoint_sha256']=self.outputs[3]['checkpoint_sha256']='new'
    def gate(self):return admit_reconstruction(self.candidate,self.outputs,self.scores,self.receipts)
    def test_completed_improvement_passes(self):self.assertTrue(self.gate()['passed'])
    def test_cost_explosion_is_rejected_despite_improved_consensus(self):
        self.outputs[2]['model_windows']=400
        self.assertFalse(self.gate()['passed'])
    def test_short_growth_cannot_replace_reconstruction_quality(self):
        self.candidate['results'][2]['metrics']['nerl']=.015
        self.assertFalse(self.gate()['passed'])
    def test_partial_or_test_results_cannot_select_weights(self):
        self.outputs[2]['all_shard_tiles_complete']=False
        with self.assertRaises(ValueError):self.gate()
        self.outputs[2]['all_shard_tiles_complete']=True;self.candidate['test_used']=True
        with self.assertRaises(ValueError):self.gate()
    def test_changed_retained_weight_is_rejected(self):
        self.outputs[0]['checkpoint_sha256']='different'
        with self.assertRaises(ValueError):self.gate()
    def test_geometry_change_is_rejected(self):
        self.outputs[2]['roi']=[0,64]*3
        with self.assertRaises(ValueError):self.gate()


if __name__=='__main__':unittest.main()
