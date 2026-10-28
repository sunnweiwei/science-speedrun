import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]/'environment/baseline/braindecode'
sys.path.insert(0,str(ROOT))
from model import Model
from data import Dataset
class SpecialistInputTests(unittest.TestCase):
    def test_cache_preserves_loader_values_and_order(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);x=np.arange(3*16*5*200,dtype=np.float64).reshape(3,16,5,200)/7
            np.save(root/'a.npy',x)
            rows=[dict(id=str(i),path='a.npy',index=j,label=i) for i,j in enumerate([2,0,1])]
            (root/'manifest.json').write_text(json.dumps({'samples':rows}))
            slow=Dataset(root);cached=Dataset(root,cache=True)
            for i in range(3):
                self.assertTrue(torch.equal(slow[i][0],cached[i][0]));self.assertEqual(slow[i][1:],cached[i][1:])
    def test_normalization_is_sample_local_and_exportable(self):
        torch.manual_seed(7)
        m=Model('tuev','eegnet','per-channel-zscore').eval()
        x=torch.randn(2,16,5,200);x[1]=0
        with torch.inference_mode():
            original=m(x);separate=m(x[:1])
            shifted=m(x*4+9)
        self.assertTrue(torch.isfinite(original).all())
        torch.testing.assert_close(original[:1],separate,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(original,shifted,atol=1e-5,rtol=1e-5)
        restored=Model('tuev','eegnet','per-channel-zscore').eval()
        restored.load_state_dict(m.state_dict())
        with torch.inference_mode():torch.testing.assert_close(original,restored(x))
        with self.assertRaises(ValueError):Model('tuev','eegnet','unknown')


    def test_groupnorm_package_preserves_inference(self):
        from types import SimpleNamespace
        from train import publish
        torch.manual_seed(7)
        model = Model('faced', 'contrawr', normalization_layer='group').eval()
        self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        self.assertTrue(any(isinstance(m, torch.nn.GroupNorm) for m in model.modules()))
        x = torch.randn(2, 32, 10, 200)
        with torch.inference_mode():
            expected = model(x)
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(output=Path(td)/'package', track='faced', model='contrawr',
                                   input_normalization='none', normalization_layer='group')
            publish(model, args, {'scope':'test'}, {})
            config = json.loads((args.output/'config.json').read_text())
            self.assertEqual(config['normalization_layer'], 'group')
            restored = Model(config['track'], config['model'], config['input_normalization'],
                             config['normalization_layer']).eval()
            restored.load_state_dict(torch.load(args.output/'weights.pt', weights_only=True))
            with torch.inference_mode():
                torch.testing.assert_close(expected, restored(x))
        with self.assertRaises(ValueError):
            Model('faced', 'contrawr', normalization_layer='invalid')
