import copy
import unittest
import numpy as np
import torch

from batched_ffn import grow_tile
from train_volume import production_windows, loss_sum
from ffn_growth_validation import admit
from model_package import ffn_inference_architecture


class OffsetModel(torch.nn.Module):
    def __init__(self, offset):
        super().__init__();self.offset=torch.nn.Parameter(torch.tensor(float(offset)))
        self.inputs=[]
    def forward(self,raw,canvas):
        self.inputs.append((raw.detach().numpy().copy(),canvas.detach().numpy().copy()))
        return canvas+self.offset


class ProductionTrainingTest(unittest.TestCase):
    def test_training_windows_match_production_grow_requests(self):
        image=np.arange(65**3,dtype=np.uint32).reshape((65,)*3).astype(np.uint8)
        raw=torch.from_numpy(image.astype(np.float32)/255)[None,None]
        target=torch.ones_like(raw);mask=torch.ones_like(raw);model=OffsetModel(6)
        gen=grow_tile(image,[16]*3,[49]*3,max_calls=8,
                      seed_provider=lambda *args:np.array([[32]*3]))
        request=next(gen)
        for logits,y,m in production_windows(model,raw,target,mask,steps=8):
            np.testing.assert_array_equal(model.inputs[-1][0][0,0],request[0])
            np.testing.assert_array_equal(model.inputs[-1][1][0,0],request[1])
            self.assertGreater(float(m.sum()),0)
            try:request=gen.send(logits.detach().numpy()[0,0])
            except StopIteration as done:self.assertTrue(done.value[1]['budget_exhausted'])

    def test_exhaustion_masks_dummy_backwards_and_keeps_gradients_finite(self):
        raw=torch.zeros((1,1,65,65,65));target=torch.ones_like(raw)
        model=OffsetModel(-7);counts=[]
        for logits,y,m in production_windows(model,raw,target,target,steps=4):
            counts.append(float(m.sum()))
            loss=loss_sum(logits,y,m)/m.sum().clamp_min(1)
            loss.backward()
        self.assertGreater(counts[0],0);self.assertEqual(counts[1:],[0.,0.,0.])
        self.assertTrue(torch.isfinite(model.offset.grad))


class GrowthAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.reference=dict(contract={'manifest_sha256':'x','shape':97},
            totals=dict(tp=100,exhausted_samples=0,dice=.8,recall=.7,precision=.95,
                        components_per_positive_voxel=.01,calls_per_correct_voxel=.1))
    def test_quality_regression_and_cost_explosion_are_rejected(self):
        c=copy.deepcopy(self.reference);c['totals'].update(dice=.6,calls_per_correct_voxel=1.)
        gate=admit(c,self.reference)
        self.assertFalse(gate['passed']);self.assertEqual(len(gate['reasons']),2)
    def test_incomplete_candidate_and_incomplete_reference_fail_closed(self):
        for key in ['candidate','reference']:
            c=copy.deepcopy(self.reference);r=copy.deepcopy(self.reference)
            (c if key=='candidate' else r)['totals']['exhausted_samples']=1
            self.assertFalse(admit(c,r)['passed'])
    def test_reference_contract_must_match(self):
        c=copy.deepcopy(self.reference);c['contract']['shape']=129
        with self.assertRaises(ValueError):admit(c,self.reference)
    def test_matching_valid_candidate_passes(self):
        self.assertTrue(admit(self.reference,self.reference)['passed'])

    def test_more_complete_growth_with_lower_cost_per_correct_voxel_passes(self):
        c=copy.deepcopy(self.reference)
        c['totals'].update(tp=140,dice=.85,recall=.8,calls_per_correct_voxel=.09)
        self.assertTrue(admit(c,self.reference)['passed'])

    def test_nonfinite_metrics_fail_closed(self):
        c=copy.deepcopy(self.reference);c['totals']['dice']=float('nan')
        with self.assertRaises(ValueError):admit(c,self.reference)

    def test_training_unroll_metadata_does_not_change_inference_architecture(self):
        a=dict(stage='ffn',width=32,ffn_recurrent_steps=4)
        b=dict(a,ffn_recurrent_steps=8)
        self.assertEqual(ffn_inference_architecture(a),ffn_inference_architecture(b))
        b['width']=64
        self.assertNotEqual(ffn_inference_architecture(a),ffn_inference_architecture(b))


class ColdStartTrainingTest(unittest.TestCase):
    def test_random_production_training_needs_no_legacy_reference(self):
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import train_volume

        class TinyCrops:
            def __init__(self,*args,**kwargs):pass
            def __len__(self):return 1
            def __getitem__(self,index):
                if index:raise IndexError(index)
                raw=torch.zeros(1,65,65,65)
                return raw,raw.clone(),torch.ones_like(raw),torch.ones_like(raw)

        growth=dict(contract={'samples':16,'shape':97,'max_calls':512},
            totals=dict(tp=100,exhausted_samples=0,dice=.8,recall=.7,precision=.95,
                        components_per_positive_voxel=.01,calls_per_correct_voxel=.1))
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);manifest=r/'manifest.json';out=r/'training'
            manifest.write_text(json.dumps({'voxel_size_nm':[16,16,16],'dataset_id':'test'}))
            argv=['train_volume.py','--manifest',str(manifest),'--output',str(out),
                  '--stage','ffn','--steps','1','--seconds','60','--device','cpu','--workers','0']
            with patch('sys.argv',argv),patch.object(train_volume,'device_guard'), \
                 patch.object(train_volume,'external',side_effect=lambda x:x), \
                 patch.object(train_volume,'validate_manifest'), \
                 patch.object(train_volume,'Crops',TinyCrops), \
                 patch.object(train_volume,'FFN15',return_value=OffsetModel(6)), \
                 patch.object(train_volume,'ffn_windows',side_effect=AssertionError('Legacy path called')), \
                 patch.object(train_volume,'growth_validation',return_value=growth) as validate:
                train_volume.main()
            validate.assert_called_once()
            run=json.loads((out/'run.json').read_text())
            self.assertEqual(run['initialization'],'random')
            self.assertEqual(run['ffn_policy'],'production')
            self.assertIsNone(run['growth_reference_sha256'])
            ckpt=torch.load(out/'best.pt',weights_only=True)
            self.assertTrue(ckpt['optimizer']['state'])
            self.assertEqual(ckpt['training_state']['completed_steps'],1)
            self.assertEqual(ckpt['training_state']['next_global_sample_index'],1)
            self.assertEqual(len(ckpt['training_state']['rng_states']),1)
            self.assertEqual(ckpt['selection_rule'],'production_seed_growth')
            self.assertTrue(ckpt['growth_admission']['passed'])


if __name__=='__main__':unittest.main()
