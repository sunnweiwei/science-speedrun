import copy
import unittest
from resume_agglomeration import resume_rows


class ResumeCandidatesTest(unittest.TestCase):
    def setUp(self):
        self.decisions=[dict(pair=[1,i],center_zyx=[50]*3) for i in (2,3,4)]
        self.signature={k:k+'-hash' for k in ('checkpoint_sha256','decisions_sha256','raw_sha256','base_sha256')}
        self.complete=dict(**self.decisions[0],complete=True,accepted=True,iou=1.,
            consistency=[[1.,1.],[1.,1.]],prediction_voxels=[10,10],deleted_voxels=[0,0])
        self.prior=dict(schema='ffn-agglomeration-shard/v1',ground_truth_used=False,
            shard_index=0,shards=4,requested_candidates=3,completed_candidates=1,
            results=[self.complete,dict(**self.decisions[1],complete=False,accepted=False,reason='time_limit')],**self.signature)

    def run_resume(self,prior):return resume_rows(prior,self.decisions,self.signature,0,4)

    def test_reuse_complete_and_retry_incomplete_and_unstarted(self):
        before=copy.deepcopy(self.prior);kept,pending=self.run_resume(self.prior)
        self.assertEqual(kept,[self.complete]);self.assertEqual(pending,self.decisions[1:])
        self.assertEqual(self.prior,before)

    def test_reject_changed_inputs_shard_and_count(self):
        for name,value in [('raw_sha256','wrong'),('checkpoint_sha256','wrong'),('base_sha256','wrong'),
                           ('decisions_sha256','wrong'),('shard_index',1),('shards',2),
                           ('completed_candidates',2),('requested_candidates',2),('ground_truth_used',True)]:
            with self.subTest(field=name):
                prior=copy.deepcopy(self.prior);prior[name]=value
                with self.assertRaises(ValueError):self.run_resume(prior)

    def test_reject_forged_candidate_results(self):
        for field,value in [('pair',[8,9]),('center_zyx',[51]*3),('complete',1),('accepted',False),('iou',0.1)]:
            with self.subTest(field=field):
                prior=copy.deepcopy(self.prior);prior['results'][0][field]=value
                with self.assertRaises(ValueError):self.run_resume(prior)
        prior=copy.deepcopy(self.prior);prior['results'].append(prior['results'][0])
        with self.assertRaises(ValueError):self.run_resume(prior)
        prior=copy.deepcopy(self.prior);prior['results'][1]['accepted']=True
        with self.assertRaises(ValueError):self.run_resume(prior)


class DiskResumeCandidatesTest(ResumeCandidatesTest):
    def setUp(self):
        super().setUp()
        import os
        from pathlib import Path
        from test_fs_support import TemporaryDirectory
        from decision_table import write_table,load_rows
        self.tmp=TemporaryDirectory(dir=os.environ['PATHFINDER_TEST_XTEMP']);self.addCleanup(self.tmp.cleanup)
        folder=Path(self.tmp.name)
        descriptor=write_table(folder/'table.npy',((tuple(r['pair']),0.,tuple(r['center_zyx'])) for r in self.decisions),3,[0,0,0])
        self.disk=load_rows(dict(decision_table=descriptor),folder/'manifest.json')
        self.addCleanup(self.disk.array._mmap.close)
        for row in self.decisions:row['distance_nm']=0.

    def run_resume(self,prior):
        from resume_agglomeration import resume_to_sink
        kept=[];counts,pending=resume_to_sink(prior,self.disk,self.signature,0,4,kept.append)
        self.assertEqual(counts['count'],len(kept));self.assertEqual(counts['accepted'],sum(r['accepted'] for r in kept))
        self.assertEqual(len(pending),len(list(pending)))
        return kept,list(pending)

    def test_strided_lookup_cannot_reuse_another_shard(self):
        selected=self.disk[::2]
        self.assertEqual(selected.pair_index([1,4]),1)
        with self.assertRaisesRegex(ValueError,'another shard'):selected.pair_index([1,3])


if __name__=='__main__':unittest.main()
