import unittest
import numpy as np
from diagnose_candidates import overlap_audit,candidate_audit,skeleton_audit


class DiagnosticTests(unittest.TestCase):
    def test_purity_unknowns_and_directed_eligibility(self):
        pred=np.ones((4,4,4),dtype=np.uint32);pred[2:]=2
        gt=np.full(pred.shape,7,dtype=np.uint16);valid=np.ones(pred.shape,bool)
        rows,_=overlap_audit(pred,gt,valid,min_known=1)
        result=candidate_audit(pred,rows,[dict(source=1,target=2,sense=.8)], [1,2],np.array([64]*3),np.array([132]*3),min_size=1)
        self.assertEqual(result['eligible_same_gt_directed_pairs'],2)
        self.assertEqual(result['same_gt_proposal_fraction_within_geometry'],.5)
        valid[2:]=False
        rows,_=overlap_audit(pred,gt,valid,min_known=1)
        self.assertFalse(rows[2]['unambiguous'])
        result=candidate_audit(pred,rows,[dict(source=1,target=2,sense=.8)], [1,2],np.array([64]*3),np.array([132]*3),min_size=1)
        self.assertEqual(result['counts'],{'ambiguous':1})

    def test_single_contaminating_node_invalidates_erl_segment(self):
        pred=np.ones((3,1,1),dtype=np.uint32)
        truth=dict(nodes=[dict(id=i,skeleton_id=1 if i<2 else 2,position_nm=[i+.5,.5,.5]) for i in range(3)],edges=[[0,1]])
        result=skeleton_audit(truth,pred,dict(origin_nm=[0]*3,voxel_size_nm=[1]*3))
        self.assertEqual(result['edge_counts'],{'merge_contaminated':1})
        self.assertEqual(result['erl_merging_segment_support'],{'1':{1:2,2:1}})


if __name__=='__main__':unittest.main()
