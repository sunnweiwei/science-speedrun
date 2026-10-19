import io
import unittest
import torch
from ffn_model import FFN15
from model_package import construct


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_preprocessing_and_weights_only_roundtrip(self):
        torch.manual_seed(42)
        model=FFN15(activation_checkpointing=False,raw_normalization=[.52,.062]).eval()
        plain=FFN15(activation_checkpointing=False).eval()
        plain.load_state_dict({k:v for k,v in model.state_dict().items() if k!='raw_normalization'},strict=True)
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        loaded=FFN15(activation_checkpointing=False).eval()
        loaded.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        raw=torch.rand(1,1,9,9,9);canvas=torch.rand_like(raw)
        with torch.no_grad():
            expected=plain((raw-model.raw_normalization[0])/model.raw_normalization[1],canvas)
            torch.testing.assert_close(model(raw,canvas),expected,rtol=0,atol=0)
            torch.testing.assert_close(loaded(raw,canvas),expected,rtol=0,atol=0)

    def test_legacy_state_and_forward_unchanged(self):
        model=FFN15(activation_checkpointing=False).eval()
        self.assertNotIn('raw_normalization',model.state_dict())
        loaded=FFN15(activation_checkpointing=False).eval();loaded.load_state_dict(model.state_dict(),strict=True)
        raw=torch.rand(1,1,9,9,9);canvas=torch.rand_like(raw)
        with torch.no_grad():
            # Original FFN forward, without optional preprocessing.
            features=model.stem(torch.cat((raw,canvas),dim=1))
            for block in model.blocks:features=block(features)
            expected=canvas+model.out(torch.relu(features))
            torch.testing.assert_close(loaded(raw,canvas),expected,rtol=0,atol=0)

    def test_invalid_and_missing_normalization_rejected(self):
        for norm in ([0,0],[0,-1],[float('nan'),1],[0,1,2]):
            with self.assertRaises(ValueError):FFN15(raw_normalization=norm)
        plain=FFN15();normalized=FFN15(raw_normalization=[0,1])
        with self.assertRaises(RuntimeError):normalized.load_state_dict(plain.state_dict(),strict=True)
        state=plain.state_dict();state['raw_normalization']=torch.tensor([0.,-1.])
        with self.assertRaises(ValueError):plain.load_state_dict(state,strict=True)

    def test_package_metadata_must_match_normalization_state(self):
        architecture=dict(stage='ffn',width=32,residual_blocks=20,input_shape=[33]*3,
                          voxel_size_nm=[16]*3,raw_normalization=[.52,.062])
        model=construct('ffn',architecture)
        model.load_state_dict(FFN15(raw_normalization=[.52,.062]).state_dict(),strict=True)
        with self.assertRaisesRegex(ValueError,'differs from configured'):
            model.load_state_dict(FFN15(raw_normalization=[.4,.1]).state_dict(),strict=True)

    def test_head_gain_changes_only_random_output_weights(self):
        from train_volume import initialize_ffn_head
        torch.manual_seed(42);plain=FFN15()
        torch.manual_seed(42);scaled=FFN15(raw_normalization=[.52,.062])
        initialize_ffn_head(scaled,3.)
        for name,value in plain.state_dict().items():
            expected=value*3 if name=='out.weight' else value
            torch.testing.assert_close(scaled.state_dict()[name],expected,rtol=0,atol=0)
        for gain in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):initialize_ffn_head(plain,gain)

    def test_random_initialization_options_reject_warm_start_and_other_stages(self):
        import train_volume
        from unittest.mock import patch
        base=['train_volume.py','--manifest','unused.json','--output','unused-output','--seconds','60','--device','cpu']
        cases=[['--stage','ffn','--ffn-head-init-gain','3','--init-checkpoint','unused.pt'],
               ['--stage','sense','--ffn-head-init-gain','3'],
               ['--stage','ffn','--ffn-raw-normalization','.52','.062','--init-checkpoint','unused.pt'],
               ['--stage','sense','--ffn-raw-normalization','.52','.062'],
               ['--stage','ffn','--ffn-raw-normalization','0','0']]
        for args in cases:
            with self.subTest(args=args),patch('sys.argv',base+args),patch.object(train_volume,'device_guard'):
                with self.assertRaises(ValueError):train_volume.main()


if __name__=='__main__':unittest.main()
