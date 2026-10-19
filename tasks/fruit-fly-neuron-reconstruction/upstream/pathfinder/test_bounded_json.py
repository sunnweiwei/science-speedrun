import json
from pathlib import Path
import unittest
from test_fs_support import TemporaryDirectory
from bounded_json import write_bounded


class BoundedJSONTests(unittest.TestCase):
    def test_exact_standard_serialization_and_one_byte_short_rejection(self):
        value={'unicode':'脑','ids':[0,2**32-1],'nested':{'x':.25}}
        expected=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        with TemporaryDirectory() as folder:
            folder=Path(folder);path=folder/'result.json'
            self.assertEqual(write_bounded(path,value,len(expected)),len(expected))
            self.assertEqual(path.read_bytes(),expected)
            failed=folder/'failed.json'
            with self.assertRaisesRegex(ValueError,'byte cap'):write_bounded(failed,value,len(expected)-1)
            self.assertFalse(failed.exists());self.assertFalse(failed.with_suffix('.json.tmp').exists())
            with self.assertRaises(ValueError):write_bounded(folder/'nan.json',{'x':float('nan')},1000)
            self.assertFalse((folder/'nan.json.tmp').exists())

    def test_existing_output_and_partial_are_never_removed(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'result.json';partial=path.with_suffix('.json.tmp');partial.write_bytes(b'evidence')
            with self.assertRaises(FileExistsError):write_bounded(path,{},100)
            self.assertEqual(partial.read_bytes(),b'evidence')
            path.write_bytes(b'complete')
            with self.assertRaises(ValueError):write_bounded(path,{},100)
            self.assertEqual(path.read_bytes(),b'complete')


if __name__=='__main__':unittest.main()
