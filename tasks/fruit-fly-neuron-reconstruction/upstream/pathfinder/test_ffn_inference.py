import copy
import os
import unittest
import numpy as np
import torch
from ffn_model import FFN15
try:
    from ffn_inference import CudaGraphFFN,Norm32,install_norm32
except ImportError:
    CudaGraphFFN=None

@unittest.skipIf(CudaGraphFFN is None,'Optional Triton/compiler dependencies unavailable')
class BackendContracts(unittest.TestCase):
    def test_training_and_cpu_rejected_before_model_mutation(self):
        model=FFN15()
        with self.assertRaisesRegex(ValueError,'eval-mode'):install_norm32(model)
        model.eval()
        with self.assertRaisesRegex(ValueError,'CUDA'):install_norm32(model)
        self.assertTrue(all(isinstance(b.norm,torch.nn.LayerNorm) for b in model.blocks))

    def test_input_contract_rejected_before_cuda_capture(self):
        predictor=CudaGraphFFN.__new__(CudaGraphFFN);predictor.max_batch=2;predictor.cache={}
        for a,b in [(np.zeros((1,33,33,33)),np.zeros((1,33,33,33))),
                    (np.zeros((0,33,33,33),np.float32),np.zeros((0,33,33,33),np.float32)),
                    (np.zeros((3,33,33,33),np.float32),np.zeros((3,33,33,33),np.float32)),
                    (np.zeros((1,32,33,33),np.float32),np.zeros((1,32,33,33),np.float32)),
                    (np.zeros((1,33,33,33),np.float32),np.zeros((2,33,33,33),np.float32))]:
            with self.assertRaisesRegex(ValueError,'Matching float32'):predictor(a,b)
        self.assertEqual(predictor.cache,{})

    @unittest.skipUnless(os.environ.get('PATHFINDER_RUN_GPU_TESTS')=='1' and bool(os.environ.get('SLURM_JOB_ID')),'Explicit allocated GPU test required')
    def test_real_ffn_changing_batch_and_output_lifetime(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        model=FFN15(activation_checkpointing=False).cuda().eval()
        fast_model=copy.deepcopy(model);original={k:v.clone() for k,v in fast_model.state_dict().items()}
        predictor=CudaGraphFFN(fast_model,max_batch=4,bf16=True)
        rng=np.random.default_rng(713);retained=[]
        for batch in (1,4,2,3,1,4):
            raw=rng.random((batch,33,33,33),dtype=np.float32)
            canvas=rng.uniform(-3,3,(batch,33,33,33)).astype(np.float32)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                expected=model(torch.from_numpy(raw[:,None]).cuda(),torch.from_numpy(canvas[:,None]).cuda())[:,0].float().cpu().numpy()
            actual=predictor(raw,canvas)
            np.testing.assert_array_equal(actual,expected)
            for old,snapshot in retained:np.testing.assert_array_equal(old,snapshot)
            retained.append((actual,actual.copy()))
        for k,v in fast_model.state_dict().items():self.assertTrue(torch.equal(v,original[k]))
        self.assertEqual(set(predictor.cache),{1,2,3,4})

if __name__=='__main__':unittest.main()
