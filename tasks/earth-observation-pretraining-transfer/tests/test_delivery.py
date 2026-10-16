"""CPU delivery regressions; no benchmark data or model training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import grade

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'environment/baseline/starter'))
import publish
from package_contract import InvalidSubmission, validate_features


class DeliveryTest(unittest.TestCase):
    def fixture(self, root):
        model = root / 'candidate'
        model.mkdir()
        for name in ['encoder.pt', 'LICENSE-OlmoEarth']:
            (model / name).write_text('fixture')
        for name in ['config.json', 'provenance.json']:
            (model / name).write_text('{}')
        (model / 'infer.py').write_text('''import argparse,h5py,numpy as np
p=argparse.ArgumentParser()
for key in ['model','input','output','device']:p.add_argument('--'+key)
a=p.parse_args()
with h5py.File(a.input) as f: ids=f['ids'].asstr()[:]
np.savez(a.output,ids=ids.astype(str),features=np.ones((len(ids),2),dtype=np.float32))
''')
        inputs = root / 'inputs.h5'
        with h5py.File(inputs, 'w') as f:
            f.create_dataset('ids', data=np.array(['a', 'b', 'c'], dtype=h5py.string_dtype()))
        return model, inputs

    def test_atomic_publication_retains_previous_and_failed_check_preserves_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, inputs = self.fixture(root)
            live = root / 'submission'
            live.mkdir()  # Same empty directory as the image.
            publish.publish(model, live, inputs)
            (model / 'encoder.pt').write_text('second')
            publish.publish(model, live, inputs)
            self.assertEqual((live / 'encoder.pt').read_text(), 'second')
            self.assertTrue(any((p / 'encoder.pt').exists() and (p / 'encoder.pt').read_text() == 'fixture'
                                for p in root.glob('.submission-previous-*')))
            (model / 'infer.py').write_text('raise RuntimeError("broken candidate")')
            with self.assertRaises(subprocess.CalledProcessError):
                publish.publish(model, live, inputs)
            self.assertEqual((live / 'encoder.pt').read_text(), 'second')

    def test_unsupported_exchange_preserves_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, inputs = self.fixture(root)
            live = root / 'submission'
            live.mkdir()
            (live / 'keep').write_text('old')
            with patch.object(publish, 'exchange', side_effect=OSError('unsupported filesystem')):
                with self.assertRaises(OSError): publish.publish(model, live, inputs)
            self.assertEqual((live / 'keep').read_text(), 'old')

    def test_invalid_features_and_missing_input_are_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, inputs = self.fixture(root)
            output = root / 'features.npz'
            for ids, values in [(['a','a','c'], np.ones((3,2),np.float32)),
                                (['a','b','c'], np.full((3,2),np.nan,np.float32)),
                                (['a','b','c'], np.ones((3,2),np.float64))]:
                np.savez(output, ids=ids, features=values)
                with self.assertRaises(InvalidSubmission): validate_features(output, inputs)
            with output.open('wb') as stream:
                np.save(stream, np.ones(3))
            with self.assertRaises(InvalidSubmission): validate_features(output, inputs)
            with self.assertRaises(FileNotFoundError): validate_features(output, root/'absent.h5')

    def test_invalid_package_zero_and_infrastructure_no_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model, _ = self.fixture(root)
            output=root/'invalid'
            command=['grade.py','--model',str(model),'--data',str(root/'missing'), '--output',str(output)]
            (model/'config.json').write_text('broken')
            with patch.object(sys,'argv',command): grade.main()
            self.assertEqual(json.loads((output/'reward.json').read_text())['reward'],0)
            (model/'config.json').write_text('{}')
            output=root/'infra'; command[-1]=str(output)
            with patch.object(sys,'argv',command), patch.object(grade.subprocess,'run',side_effect=FileNotFoundError('bwrap')):
                with self.assertRaises(FileNotFoundError): grade.main()
            self.assertFalse((output/'reward.json').exists())

    def test_replay_invalid_output_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model,_=self.fixture(root); output=root/'score'
            command=['grade.py','--model',str(model),'--data',str(root), '--output',str(output)]
            with patch.object(sys,'argv',command), patch.object(grade.subprocess,'run',return_value=subprocess.CompletedProcess([],42)):
                grade.main()
            self.assertEqual(json.loads((output/'report.json').read_text())['status'],'invalid_submission')
            self.assertEqual(json.loads((output/'reward.json').read_text()),{'reward':0.0,'invalid_submission':1})


if __name__ == '__main__': unittest.main()
