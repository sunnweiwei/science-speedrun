import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import hashlib
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'environment/evaluation'))
from bci_score import score,read_predictions
spec=importlib.util.spec_from_file_location('materialize',ROOT/'environment/data/materialize.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
validator_spec=importlib.util.spec_from_file_location('validate_runtime',ROOT/'environment/data/validate_runtime.py')
validator=importlib.util.module_from_spec(validator_spec);validator_spec.loader.exec_module(validator)
class ProtocolTests(unittest.TestCase):
    def test_split_is_subject_disjoint_and_repeatable(self):
        labels={str(i):i%3 for i in range(90)}
        a=m.subject_split(labels,labels);self.assertEqual(a,m.subject_split(labels,labels))
        self.assertEqual(len(set(sum(a.values(),[]))),90)
        for subs in a.values():self.assertEqual({labels[s] for s in subs},{0,1,2})
    def test_ad_patient_weight_not_window_weight(self):
        rows=[{'id':str(i),'subject':'a','label':0} for i in range(9)]
        rows += [{'id':'9','subject':'b','label':1},{'id':'10','subject':'c','label':2}]
        p=np.tile([.8,.1,.1],(11,1));p[9]=[.1,.8,.1];p[10]=[.1,.1,.8]
        p[0]=[.1,.8,.1]
        result=score('ad65',rows,p);self.assertEqual(result['reward'],1);self.assertLess(result['balanced_accuracy'],1)
    def test_session_subject_macro(self):
        rows=[{'id':str(i),'subject':s,'session':ses,'label':y} for i,(s,ses,y) in enumerate([('a','1',0),('a','1',1),('a','2',0),('a','2',1),('b','1',0),('b','1',1)])]
        p=np.array([[1,0],[0,1],[1,0],[1,0],[1,0],[0,1]],float)
        self.assertAlmostEqual(score('longitudinal',rows,p)['reward'],.875)
    def test_prediction_contract(self):
        truth=[{'id':'a'},{'id':'b'}]
        valid=[{'id':'b','probabilities':[0,1]},{'id':'a','probabilities':[1,0]}]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'p.jsonl'
            def put(rows):p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            put(valid);np.testing.assert_array_equal(read_predictions(p,truth,2),[[1,0],[0,1]])
            bad=[valid[:1],valid+[valid[0]],[{'id':'a','probabilities':[float('nan'),0]},valid[0]],[{'id':'a','probabilities':[1,0,0]},valid[0]],[{'id':'a','probabilities':[.2,.2]},valid[0]]]
            for rows in bad:
                put(rows)
                with self.assertRaises(ValueError):read_predictions(p,truth,2)
    def test_private_writer_no_labels_or_source_in_public(self):
        with tempfile.TemporaryDirectory() as td:
            w=m.Writer(Path(td),'tuev','test')
            x=np.arange(16000,dtype=np.float64).reshape(16,5,200)
            w.add(x,3,'class_named_source.pkl','subject1');w.finish()
            manifest=json.loads((w.root/'manifest.json').read_text())
            self.assertEqual(set(manifest['samples'][0]),{'id','path','index'})
            a=np.load(w.root/manifest['samples'][0]['path'])
            np.testing.assert_array_equal(a[0]/100,x/100)
            self.assertTrue((Path(td)/'private/tuev/truth.jsonl').is_file())
            self.assertEqual((Path(td)/'private').stat().st_mode & 0o777, 0o700)
            self.assertEqual((Path(td)/'private/tuev/truth.jsonl').stat().st_mode & 0o777, 0o600)
    def test_duplicate_source_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            w=m.Writer(Path(td),'ad65','train');x=np.zeros((19,10,200))
            w.add(x,0,'one','s')
            with self.assertRaises(ValueError):w.add(x,0,'one','s')
    def test_prepared_asset_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);w=m.Writer(root,'tuev','test')
            w.add(np.zeros((16,5,200)),0,'source','subject');w.finish()
            self.assertEqual(validator.validate_split(root,'tuev','test',1)['samples'],1)
            shard=next((root/'private/tuev/public').glob('*.npy'))
            with shard.open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(ValueError):validator.validate_split(root,'tuev','test',1)
    def test_tuev_supported_source_layouts(self):
        m.validate_tuev_layout((66050,17882,29421))
        m.validate_tuev_layout((68445,15487,29421))
        m.validate_tuev_layout((83932,0,29421))
        with self.assertRaises(ValueError):m.validate_tuev_layout((66050,15487,29421))

    def test_tuev_layouts_have_identical_frozen_subject_assignment(self):
        subjects = [f'{i:08d}' for i in range(290)]
        names = []
        for i, subject in enumerate(subjects):
            count = (295 + (i < 5)) if i < 232 else (267 + (i == 232))
            names.extend(f'{subject}_{j:05d}.pkl' for j in range(count))
        tests = [f'test_{i:05d}.pkl' for i in range(29421)]
        union = hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest()
        membership = hashlib.sha256('\n'.join(subjects).encode()).hexdigest()
        def plan(boundary):
            groups = {'processed_train': names[:boundary],
                      'processed_eval': names[boundary:], 'processed_test': tests}
            def glob(path, pattern):
                return [path / name for name in groups[path.name]]
            with patch.object(Path, 'glob', glob), \
                 patch.object(m, 'TUEV_DEVELOPMENT_SHA256', union), \
                 patch.object(m, 'TUEV_SUBJECTS_SHA256', membership):
                rows, splits = m.tuev_plan(Path('/fixture'))
            return [(p.name, split, subject) for p, split, subject in rows], splits
        old, old_splits = plan(66050)
        current, current_splits = plan(68445)
        self.assertEqual(old, current)
        self.assertEqual(old_splits, current_splits)
        combined, combined_splits = plan(83932)
        self.assertEqual(old, combined)
        self.assertEqual(old_splits, combined_splits)
        self.assertFalse(set(old_splits['train']) & set(old_splits['val']))
        self.assertEqual(sum(split == 'test' for _, split, _ in old), 29421)
        original = names[0]
        names[0] = names[1]
        with self.assertRaises(ValueError):
            plan(66050)
        names[0] = original.replace('.pkl', '_changed.pkl')
        with self.assertRaises(ValueError):
            plan(68445)
if __name__=='__main__':unittest.main()
