import unittest
from diagnose_reconstruction import diagnose


class PairedDiagnosticTests(unittest.TestCase):
    def fixture(self):
        return dict(nodes=[dict(id=i,skeleton_id=1 if i<3 else 2,position_nm=[i*10,0,0]) for i in range(5)],edges=[[0,1],[1,2],[3,4]])
    def test_restored_edge_can_be_harmful_and_coverage_is_explicit(self):
        truth=self.fixture();before=[10,20,0,30,30];after=[10,10,0,10,10]
        r=diagnose(truth,before,after,processed_ids=[10],scored_ids=[10,20],merged_ids=[10,20,30])
        self.assertEqual(len(r['restored_edges']),1)
        self.assertTrue(r['restored_edges'][0]['visible_mixed_segment'])
        self.assertEqual(r['per_skeleton'][0]['restored_in_visible_mixed_segment_nm'],10.)
        self.assertEqual(r['per_skeleton'][0]['sense_processed_nodes'],1)
        self.assertEqual(r['visible_mixed_segments_before'],{})
        self.assertEqual(r['visible_mixed_segments_after'],{10:[1,2]})
        identity=diagnose(truth,before,before)
        self.assertEqual(identity['restored_edges'],[])
    def test_rejects_omission_changes_and_splits(self):
        truth=self.fixture()
        with self.assertRaises(ValueError):diagnose(truth,[10,20,0,30,30],[10,20,40,30,30])
        with self.assertRaises(ValueError):diagnose(truth,[10,10,0,30,30],[10,20,0,30,30])

if __name__=='__main__':unittest.main()
