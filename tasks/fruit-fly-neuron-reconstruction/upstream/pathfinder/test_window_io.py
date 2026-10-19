import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from window_io import save_compressed,window_array,verify_compressed


class WindowIOTests(unittest.TestCase):
    def test_stronger_compression_preserves_identical_npy_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            a=np.tile(np.arange(256,dtype=np.uint32),(32,16,1))
            receipts=[save_compressed(a,Path(folder)/f'{level}.npz',600000,compression_level=level) for level in (1,6)]
            self.assertEqual(receipts[0]['sha256'],receipts[1]['sha256'])
            for r in receipts:
                with window_array(r) as restored:np.testing.assert_array_equal(restored,a)
            for level in (-1,10,True):
                with self.assertRaisesRegex(ValueError,'compression level'):
                    save_compressed(a,Path(folder)/'invalid.npz',600000,compression_level=level)

    def test_exact_npy_round_trip_including_large_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            a=np.random.default_rng(3).integers(0,2**32,(7,9,11),dtype=np.uint32)
            original=io.BytesIO();np.save(original,a,allow_pickle=False)
            path=Path(folder)/'window.npz';r=save_compressed(a,path,10000)
            self.assertEqual(r['sha256'],hashlib.sha256(original.getvalue()).hexdigest())
            self.assertLessEqual(path.stat().st_size,10000)
            with window_array(r) as restored:np.testing.assert_array_equal(restored,a)
            corrupt=copy.deepcopy(r);corrupt['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'hash'):
                with window_array(corrupt):pass
            corrupt=copy.deepcopy(r);corrupt['shape']=[1,1,1]
            with self.assertRaises(ValueError):
                with window_array(corrupt):pass

    def test_compression_limit_failure_leaves_no_published_or_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'window.npz';a=np.random.default_rng(9).integers(0,2**32,(32,32,32),dtype=np.uint32)
            with self.assertRaisesRegex(ValueError,'reserved bytes'):save_compressed(a,path,1024)
            self.assertFalse(path.exists());self.assertFalse(path.with_suffix('.npz.partial').exists())

    def test_streaming_verifier_checks_identity_shape_and_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            a=np.arange(60,dtype=np.uint32).reshape(3,4,5)
            r=save_compressed(a,Path(folder)/'window.npz',10000)
            for chunk in (1,17,8192):
                result=verify_compressed(r,chunk_bytes=chunk)
                self.assertEqual(result['sha256'],r['sha256'])
                self.assertEqual(result['uncompressed_bytes'],r['uncompressed_bytes'])
            for key,value in [('sha256','0'*64),('archive_sha256','0'*64),('shape',[5,4,3]),('uncompressed_bytes',100)]:
                with self.assertRaises(ValueError):verify_compressed(dict(r,**{key:value}))
            for chunk in (0,True,64*1024*1024+1):
                with self.assertRaises(ValueError):verify_compressed(r,chunk_bytes=chunk)

    def test_failed_stream_verification_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'window.npz'
            with patch('window_io.verify_compressed',side_effect=ValueError('verification failed')):
                with self.assertRaisesRegex(ValueError,'verification failed'):
                    save_compressed(np.zeros((3,4,5),dtype=np.uint32),path,10000)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.npz.partial').exists())

    def test_raw_npy_adapter_preserves_existing_hash_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'window.npy';a=np.arange(60,dtype=np.uint32).reshape(3,4,5);np.save(path,a)
            with window_array(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())) as restored:np.testing.assert_array_equal(restored,a)

if __name__=='__main__':unittest.main()
