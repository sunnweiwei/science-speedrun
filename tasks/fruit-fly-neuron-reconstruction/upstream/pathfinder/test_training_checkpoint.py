from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from train_volume import atomic_checkpoint


class TrainingCheckpointTests(unittest.TestCase):
    def test_failed_serialization_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt'
            atomic_checkpoint({'step':1},path)
            before=path.read_bytes()
            def fail(checkpoint,temporary):
                temporary.write_bytes(b'incomplete')
                raise OSError('simulated disk interruption')
            with patch('train_volume.torch.save',side_effect=fail):
                with self.assertRaises(OSError):atomic_checkpoint({'step':2},path)
            self.assertEqual(path.read_bytes(),before)
            atomic_checkpoint({'step':2},path)
            self.assertEqual(torch.load(path,weights_only=True)['step'],2)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_saved_optimizer_reproduces_next_update(self):
        torch.manual_seed(2);model=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
        def update(model,optimizer):
            optimizer.zero_grad(set_to_none=True)
            model(torch.tensor([[.2,.7]])).square().sum().backward()
            optimizer.step()
        update(model,optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt'
            atomic_checkpoint(dict(model=model.state_dict(),optimizer=optimizer.state_dict()),path)
            state=torch.load(path,weights_only=True)
        restored=torch.nn.Linear(2,1);restored.load_state_dict(state['model'])
        resumed=torch.optim.AdamW(restored.parameters(),lr=.003);resumed.load_state_dict(state['optimizer'])
        update(model,optimizer);update(restored,resumed)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[name],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
