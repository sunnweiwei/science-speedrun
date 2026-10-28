import importlib.util,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=module('private_layout_materialize',ROOT/'environment/data/materialize.py')
repack=module('private_layout_repack',ROOT/'environment/data/repack_private.py')
LABELS=np.repeat([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8],3)
def attack(root):
    pub=json.loads((root/'public/manifest.json').read_text())['samples']
    truth={r['id']:r for r in map(json.loads,(root/'truth.jsonl').read_text().splitlines())}
    y=[truth[r['id']]['label'] for r in pub]
    pred=[int(LABELS[(int(Path(r['path']).stem.split('-')[-1])*128+r['index'])%84]) for r in pub]
    return balanced_accuracy_score(y,pred)

class PrivateLayoutTests(unittest.TestCase):
    def test_global_shuffle_breaks_metadata_attack_and_preserves_alignment(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);w=m.Writer(root,'faced','test')
            expected={}
            for i,label in enumerate(np.tile(LABELS,2)):
                x=np.zeros((32,10,200),np.float32)
                w.add(x,int(label),str(i),str(i//84))
                expected[str(i)]=(x,int(label))
            self.assertFalse(list(w.root.glob('*.npy')),'test data must not flush in source order')
            with patch.object(m.secrets,'SystemRandom',return_value=random.Random(3407)):w.finish()
            base=root/'private/faced'
            self.assertLess(attack(base),.25)
            truth=list(map(json.loads,(base/'truth.jsonl').read_text().splitlines()))
            for row in truth:
                x,y=expected[row['source']]
                np.testing.assert_array_equal(np.load(w.root/row['path'])[row['index']],x)
                self.assertEqual(row['label'],y)
            # Nonzero signals detect accidental signal/label desynchronization.
            second=m.Writer(root,'tuev','test')
            for i in range(260):second.add(np.full((16,5,200),i,np.float32),i%6,str(i),'s')
            with patch.object(m.secrets,'SystemRandom',return_value=random.Random(17)):second.finish()
            target=root/'repacked-tuev'
            result=repack.repack(root/'private/tuev',target)
            self.assertEqual(result['samples'],260)
            for row in map(json.loads,(target/'truth.jsonl').read_text().splitlines()):
                self.assertEqual(row['label'],int(row['source'])%6)
                self.assertTrue((np.load(target/'public'/row['path'])[row['index']]==int(row['source'])).all())

    def test_old_layout_attack_is_perfect(self):
        y=np.tile(LABELS,2)
        self.assertEqual(balanced_accuracy_score(y,[LABELS[i%84] for i in range(168)]),1.)

