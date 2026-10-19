import json
from pathlib import Path
from test_fs_support import TemporaryDirectory
import unittest
from unittest.mock import patch
from run_continuous_postprocess import plan


class PlanTests(unittest.TestCase):
    @patch("model_package.verify", return_value={"verified": True})
    def test_explicit_data_replaces_frozen_directory_and_preserves_physical_origin(self, verify):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'alternate-image.json';image.write_text('{}')
            assembly=root/'alternate-prediction';assembly.mkdir()
            (assembly/'metadata.json').write_text(json.dumps(dict(axes='zyx',voxel_size_nm=[16]*3,origin_nm=[100,200,300])))
            for name in ('receipt.json','instance-index.json','coverage.json','segmentation.npy'):(assembly/name).touch()
            (assembly/'receipt.json').write_text('{}')
            package=root/'model-package'
            for stage in ('sense','shape'):
                p=package/f'weights/{stage}.pt';p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            result=plan(root,root/'out',10,20,30,image=image,assembled=assembly,package=package)
            command=result['shape'][2]
            start=command.index('--origin-nm')
            self.assertEqual(command[start+1:start+4],['100','200','300'])
            for command in result['sense']:
                self.assertIn(str(image),command);self.assertIn(str(assembly),command)
                self.assertNotIn(str(root/'continuous24/manifest.json'),command)
            (assembly/'coverage.json').unlink()
            with self.assertRaises(FileNotFoundError):plan(root,root/'out',10,20,30,image=image,assembled=assembly,package=package)

    def test_implicit_historical_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError,'Explicit'):
            plan(Path('/unused'),Path('/unused/out'),10,20,30)

if __name__=='__main__':unittest.main()
