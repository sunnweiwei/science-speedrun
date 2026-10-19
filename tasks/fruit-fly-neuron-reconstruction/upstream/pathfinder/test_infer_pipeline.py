import copy
import unittest
import numpy as np
from infer_pipeline import candidate_pass, validate_geometry


class CandidatePipelineTest(unittest.TestCase):
    def fixture(self):
        raw=np.zeros((160,160,160),np.uint8)
        seg=np.zeros((32,32,32),np.uint32)
        seg[8:16,8:16,8:16]=1;seg[16:24,16:24,16:24]=2
        receipt=dict(ground_truth_used=False,roi=[64,96]*3,budget_exhausted=False,
            metadata=dict(axes='zyx',voxel_size_nm=[16]*3,origin_nm=[924,1024,1124],
            raw_origin_nm=[-100,0,100],shape=[32]*3,coordinate_convention='voxel_outer_corner'))
        return raw,seg,receipt

    def test_predicted_instances_produce_bidirectional_edges_at_full_fov(self):
        raw,seg,receipt=self.fixture()
        shapes=[]
        def predict(crop,prompt):
            shapes.append(crop.shape)
            self.assertEqual(prompt[0,0,0],.5)
            self.assertEqual(crop.dtype,np.float32)
            return np.ones(crop.shape,np.float32)
        result,audit=candidate_pass(raw,seg,receipt,predict,seconds=10,max_calls=2)
        self.assertEqual(shapes,[(128,128,128)]*2)
        self.assertEqual({(r['source'],r['target'],r['sense']) for r in result['edges']},{(1,2,1.),(2,1,1.)})
        self.assertEqual(result['metadata']['origin_nm'],[924,1024,1124])
        self.assertFalse(audit['budget_exhausted'])
        self.assertEqual(audit['instance_coverage'],1.)

    def test_zero_budget_and_empty_segmentation_are_explicit(self):
        raw,seg,receipt=self.fixture()
        def never(*args): self.fail('predict must not be called')
        result,audit=candidate_pass(raw,seg,receipt,never,seconds=10,max_calls=0)
        self.assertTrue(audit['budget_exhausted']);self.assertEqual(audit['unprocessed_instance_ids'],[1,2])
        seg[:]=0
        result,audit=candidate_pass(raw,seg,receipt,never,seconds=10,max_calls=10)
        self.assertTrue(audit['empty_segmentation']);self.assertEqual(result['edges'],[])

    def test_boundary_context_not_padded_or_shrunk(self):
        raw,seg,receipt=self.fixture()
        receipt['roi']=[0,32]*3
        receipt['metadata']['origin_nm']=[-100,0,100]
        result,audit=candidate_pass(raw,seg,receipt,lambda *args:self.fail('boundary call'),seconds=10,max_calls=10)
        self.assertEqual(audit['skipped_boundary_instance_ids'],[1,2]);self.assertFalse(audit['budget_exhausted'])

    def test_invalid_coordinates_and_gt_receipt_rejected(self):
        raw,seg,receipt=self.fixture()
        receipt['metadata']['origin_nm'][0]+=16
        with self.assertRaisesRegex(ValueError,'metadata'):validate_geometry(raw,seg,receipt)
        receipt['ground_truth_used']=True
        with self.assertRaisesRegex(ValueError,'EM-only'):validate_geometry(raw,seg,receipt)

    def test_full_segment_denominator_counts_outside_crop(self):
        raw=np.zeros((256,256,256),np.uint8)
        seg=np.zeros(raw.shape,np.uint32)
        seg[127:129,127:129,127:129]=1
        seg[96:160,96:160,130:134]=2
        seg[:64,:64,:64]=2
        receipt=dict(ground_truth_used=False,roi=[0,256]*3,metadata=dict(axes='zyx',voxel_size_nm=[16]*3,
            origin_nm=[0]*3,raw_origin_nm=[0]*3,shape=[256]*3,coordinate_convention='voxel_outer_corner'))
        result,audit=candidate_pass(raw,seg,receipt,lambda x,p:np.ones(x.shape,np.float32),seconds=30,max_calls=1,min_size=1)
        self.assertEqual(result['edges'],[])
        self.assertGreater(audit['full_sizes'][2],128**3//10)


if __name__=='__main__':unittest.main()
