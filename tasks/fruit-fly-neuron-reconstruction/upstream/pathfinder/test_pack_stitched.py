import json
import os
from pathlib import Path
import unittest
import numpy as np
from pack_stitched import run
from window_io import window_array
from stitch_ffn import run as stitch
from test_stitch_ffn import fixture,xtemp_directory
from common import sha256


@unittest.skipUnless(all(os.environ.get(k) for k in ('PATHFINDER_TEST_XTEMP','PATHFINDER_TEST_FFN_SOURCE','PATHFINDER_TEST_CONNECTOMICS_SOURCE')),'Xtemp and author sources required')
class PackStitchedTests(unittest.TestCase):
    def test_package_preserves_original_array_and_complete_stitch_provenance(self):
        with xtemp_directory() as folder:
            root=Path(folder);manifest=fixture(root);path=root/'manifest.json';path.write_text(json.dumps(manifest))
            output=root/'stitched'
            stitch(path,root,output,Path(os.environ['PATHFINDER_TEST_FFN_SOURCE']),Path(os.environ['PATHFINDER_TEST_CONNECTOMICS_SOURCE']),1)
            source=output/'segmentation.npy';original=sha256(source)
            with self.assertRaisesRegex(ValueError,'memory'):run(output,path,root,root/'too-small',100000,.001)
            self.assertFalse((root/'too-small').exists())
            result=run(output,path,root,root/'package',100000,2)
            self.assertTrue(result['complete']);self.assertTrue(result['original_retained'])
            self.assertEqual(original,sha256(source));self.assertEqual(original,result['storage']['sha256'])
            with window_array(result['storage']) as restored:np.testing.assert_array_equal(restored,np.load(source))
            with self.assertRaisesRegex(ValueError,'reserved bytes'):run(output,path,root,root/'cap-failure',1,2)
            self.assertFalse((root/'cap-failure/receipt.json').exists())
            self.assertEqual(original,sha256(source))
            bad=json.loads((output/'receipt.json').read_text());bad['complete']=False;(output/'receipt.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'Complete'):run(output,path,root,root/'partial',100000,2)
            self.assertFalse((root/'partial').exists())

if __name__=='__main__':unittest.main()
