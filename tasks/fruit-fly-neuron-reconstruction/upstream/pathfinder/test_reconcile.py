import unittest
import numpy as np
from index_storage import NumericNodes
from reconcile import reconcile,remap_slabs,streaming_nodes,indexed_nodes


def edge(a,b,s):return dict(source=a,target=b,sense=s,shape=.9)


class ReconcileTests(unittest.TestCase):
    def test_global_objective_beats_first_edge(self):
        # The highest single edge blocks two better combined edges.
        edges=[edge(1,2,.9),edge(1,3,.6),edge(2,4,.6)]
        motifs=[((1,2),.9),((1,3),.9),((2,4),.9),((1,2,3),0.),((1,2,4),0.),((1,2,3,4),0.)]
        result=reconcile([1,2,3,4],edges,motifs)
        self.assertEqual(result['components'],[[1,3],[2,4]])
    def test_unscored_motif_prevents_merge(self):
        r=reconcile([1,2,3],[edge(1,2,.8),edge(2,3,.7)],[((1,2),.9),((2,3),.9)])
        self.assertEqual(r['components'],[[1,2],[3]])
        self.assertEqual(r['missing_motif_scores'],1)
    def test_background_no_edges_identity(self):
        self.assertEqual(reconcile([1,2],[],[])['components'],[[1],[2]])
    def test_indirect_forbidden_motif(self):
        motifs=[((1,2),.9),((2,3),.9),((1,2,3),.9),((1,3),0.)]
        self.assertNotEqual(reconcile([1,2,3],[edge(1,2,.8),edge(2,3,.7)],motifs)['components'],[[1,2,3]])

    def test_many_isolates_do_not_enter_search(self):
        r=reconcile(list(range(1,100001)),[edge(1,2,.9)],[((1,2),.9)],include_singletons=False)
        self.assertEqual(r['components'],[[1,2]]);self.assertEqual(len(r['methods']),1)
        self.assertEqual(r['singleton_count'],99998)
    def test_numeric_nodes_match_legacy_search_with_sparse_ids_and_isolates(self):
        ids=[1,2,3,17,4000000000];numeric=NumericNodes(np.asarray(ids,np.uint64))
        edges=[edge(1,2,.8),edge(2,3,.7)];motifs=[((1,2),.9),((2,3),.9)]
        for singletons in (False,True):
            self.assertEqual(reconcile(numeric,edges,motifs,include_singletons=singletons),
                             reconcile(ids,edges,motifs,include_singletons=singletons))
        for bad in ([1,1],[2,1],[0,1]):
            with self.assertRaises(ValueError):NumericNodes(np.asarray(bad,np.uint64))
        with self.assertRaisesRegex(ValueError,'Unknown'):reconcile(numeric,[edge(1,9,.9)],[])
        with self.assertRaisesRegex(ValueError,'motif'):reconcile(numeric,[],[((1,9),.9)])

    def test_sparse_uint32_remap_preserves_background_and_isolates(self):
        seg=np.array([0,1,4_000_000_000,7,0,4_000_000_000,99,1],dtype=np.uint32).reshape(2,2,2)
        out=np.empty_like(seg);count=remap_slabs(seg,out,[[1,4_000_000_000]],slab_depth=1)
        expected=seg.copy();expected[expected==4_000_000_000]=1
        np.testing.assert_array_equal(out,expected);self.assertEqual(count,1)
        self.assertEqual(streaming_nodes(seg,1),[1,7,99,4_000_000_000])
    def test_index_requires_matching_provenance(self):
        index=dict(schema='predicted-instance-index/v1',axes='zyx',shape=[2,3,4],ground_truth_used=False,
                   segmentation_sha256='abc',records=[dict(id=1),dict(id=4_000_000_000)])
        self.assertEqual(indexed_nodes(index,(2,3,4),'abc'),[1,4_000_000_000])
        for change in [dict(ground_truth_used=True),dict(segmentation_sha256='wrong'),dict(shape=[3,3,4])]:
            with self.assertRaises(ValueError):indexed_nodes(dict(index,**change),(2,3,4),'abc')
    def test_missing_motifs_remain_forbidden_at_zero_threshold_greedy(self):
        r=reconcile([1,2,3],[edge(1,2,.9),edge(2,3,.8)],[((1,2),.9),((2,3),.9)],threshold=0,max_exact_edges=0)
        self.assertEqual(r['components'],[[1,2],[3]])

if __name__=='__main__':unittest.main()
