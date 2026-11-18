import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'protocol'))
try:
    import torch
    from encoder_checkpoint import load_model_weights
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "The runtime tests require PyTorch")
class EncoderCheckpointTests(unittest.TestCase):
    def test_loads_current_and_non_tar_legacy_tensor_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"checkpoint.pt"
            value=torch.tensor([1.5,2.5],dtype=torch.float64)
            for zip_format in (True,False):
                torch.save({"model":{"weight":value},"epoch":5},path,
                           _use_new_zipfile_serialization=zip_format)
                actual=load_model_weights(path)["weight"]
                self.assertEqual(actual.dtype,value.dtype)
                self.assertTrue(torch.equal(actual,value))

    def test_rejects_torchscript_and_legacy_tar_before_native_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"checkpoint.pt"
            with zipfile.ZipFile(p,"w") as archive:
                archive.writestr("archive/constants.pkl",b"")
            with self.assertRaisesRegex(ValueError,"TorchScript"):
                load_model_weights(p)
            with tarfile.open(p,"w") as archive:
                item=tarfile.TarInfo("pickle")
                archive.addfile(item,io.BytesIO())
            with self.assertRaisesRegex(ValueError,"TAR"):
                load_model_weights(p)

    def test_checkpoint_reducer_is_not_executed(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"checkpoint.pt"
            flag=Path(temp)/"executed"
            class Exploit:
                def __reduce__(self):
                    return (eval,(f"open({str(flag)!r}, 'w').close()",))
            torch.save({"model":{},"unused":Exploit()},p)
            import pickle
            with self.assertRaises(pickle.UnpicklingError):
                load_model_weights(p)
            self.assertFalse(flag.exists())
