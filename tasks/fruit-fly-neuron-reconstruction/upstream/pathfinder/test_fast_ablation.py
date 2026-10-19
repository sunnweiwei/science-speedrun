import unittest
from reconcile import reconcile_pairs
from shared_proposals import build


class FastAblationTests(unittest.TestCase):
    def test_pair_matching_is_deterministic_and_prevents_unscored_triples(self):
        edges=[dict(source=a,target=b,sense=.9,shape=s) for a,b,s in
               [(1,2,.95),(2,3,.96),(4,5,.7),(6,7,.8)]]
        motifs=[([e['source'],e['target']],e['shape']) for e in edges]
        out=reconcile_pairs(list(range(1,9)),edges,motifs)
        self.assertEqual(out['components'],[[2,3],[6,7]])
        self.assertEqual(out['singleton_count'],4)
        self.assertEqual(out,reconcile_pairs(list(range(1,9)),edges[::-1],motifs[::-1]))
        self.assertFalse(out['transitive_merges_allowed'])
        with self.assertRaises(ValueError):reconcile_pairs([1,2],edges[:1],[])

    def test_shared_candidates_require_complete_prediction_only_coverage(self):
        obj=dict(ground_truth_used=False,provenance=dict(segmentation_sha256='base'),
                 coverage=dict(unprocessed_instance_ids=[],processed_count=2,eligible_count=2),
                 edges=[dict(source=2,target=1,sense=.9,center_zyx=[100,100,100])])
        out=build(obj,'base',[200]*3)
        self.assertEqual(out['decisions'],[dict(pair=[1,2],center_zyx=[100]*3)])
        self.assertFalse(out['independent_watershed_decisions'])
        obj['coverage']['unprocessed_instance_ids']=[2]
        with self.assertRaises(ValueError):build(obj,'base',[200]*3)
        obj['coverage']['unprocessed_instance_ids']=[]
        with self.assertRaises(ValueError):build(obj,'changed',[200]*3)

    def test_recenter_keeps_distant_proposed_target_inside_ffn_analysis(self):
        import numpy as np
        base=np.zeros((200,)*3,np.uint32);base[100,100,100]=1;base[160,100,100]=2
        obj=dict(ground_truth_used=False,provenance=dict(segmentation_sha256='base'),
            coverage=dict(unprocessed_instance_ids=[],processed_count=1,eligible_count=1),
            edges=[dict(source=1,target=2,sense=.9,center_zyx=[100]*3,evidence_seed_id=1)])
        result=build(obj,'base',base.shape,base)
        self.assertEqual(result['decisions'][0]['center_zyx'],[130,100,100])
        base[160,100,100]=0
        with self.assertRaises(ValueError):build(obj,'base',base.shape,base)


if __name__=='__main__':unittest.main()
