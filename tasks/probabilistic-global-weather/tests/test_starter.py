import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment'))
from build_starter import build

ROOT=Path(__file__).resolve().parents[1]

class StarterTest(unittest.TestCase):
    def test_delivered_boundary_and_runtime_functions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'agent';build(ROOT,output);starter=output/'starter'
            excluded={'collect_profiles.py','profile_reuse.py','verify_track.py','verify_case.py','test.sh','baseline.py','audit_data.py'}
            self.assertFalse(excluded & {p.name for p in starter.iterdir()})
            self.assertFalse(list(starter.glob('test_*')))
            self.assertFalse((output/'evaluation/verify_track.py').exists())
            self.assertTrue((output/'evaluation/development.py').exists())
            self.assertTrue((starter/'ar_extension.py').exists())
            self.assertTrue((starter/'inference_state.py').exists())
            self.assertIn('inference_state.py',(starter/'package_model.py').read_text())
            self.assertIn('wn2-inference/v3',(starter/'package_model.py').read_text())
            schedule=json.loads((starter/'schedule.json').read_text())
            self.assertEqual([s['short_updates'] for s in schedule['stages']], [410,102,32,8,4,4,4,4,4,4,4])
            self.assertEqual(schedule['training_cap_seconds'],86400)
            self.assertEqual(schedule['training_cap_seconds']+schedule['checkpoint_reserve_seconds'],88200)
            for stage in schedule['stages'][5:]:
                self.assertAlmostEqual(stage['warmup_equivalent_updates'],.4)
            for p in starter.glob('*.py'):compile(p.read_text(),str(p),'exec')
            def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
            original=functions(ROOT/'environment/weathernext/wn2/train_stage.py');delivered=functions(starter/'train_stage.py')
            for name in ['run','make_functions','optimizer','draw_indices']:self.assertEqual(original[name],delivered[name])
            self.assertNotIn("'--profile'",(starter/'train_stage.py').read_text())
            self.assertNotIn('def plan(', (starter/'plan_schedule.py').read_text())
            self.assertNotIn('def run_probe(', (starter/'wn2_distributed.py').read_text())
            self.assertNotIn('def main(', (starter/'wn2_validation.py').read_text())

    def test_complete_run_has_eleven_stages_and_export_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build(ROOT,root/'agent');source=root/'agent/starter/train.py'
            spec=importlib.util.spec_from_file_location('starter_train',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            calls=[]
            def execute(cmd,**kwargs):
                calls.append(cmd)
                self.assertGreater(kwargs['timeout'],0)
                self.assertLessEqual(kwargs['timeout'],88200)
                if Path(cmd[1]).name=='train_stage.py':
                    output=Path(cmd[cmd.index('--output')+1]);(output/'report.json').write_text('{"status":"passed"}')
                    (output/'state.npz').write_bytes(b'checkpoint fixture')
                if Path(cmd[1]).name=='package_model.py':
                    output=Path(cmd[cmd.index('--output')+1]);output.mkdir();(output/'infer.sh').write_text('#!/bin/bash\n')
            (root/'submission').mkdir();(root/'submission'/'old.txt').write_text('previous package')
            args=['train.py','--data',str(root/'data'),'--output',str(root/'run'),'--submission',str(root/'submission'),'--seed','7']
            with patch('sys.argv',args),patch.object(module.subprocess,'run',side_effect=execute):module.main()
            self.assertEqual(len(calls),12)
            for index,cmd in enumerate(calls[:11]):
                self.assertEqual(cmd[cmd.index('--stage-index')+1],str(index))
                self.assertEqual(cmd[cmd.index('--seed')+1],'7')
                self.assertNotIn('--profile',cmd)
                self.assertEqual('--resume' in cmd,index>0)
            self.assertEqual(Path(calls[-1][1]).name,'package_model.py')
            # Packaged next to the submission, then swapped in whole: the old package is gone.
            self.assertNotEqual(Path(calls[-1][calls[-1].index('--output')+1]),root/'submission')
            self.assertEqual(sorted(p.name for p in (root/'submission').iterdir()),['infer.sh'])
            self.assertFalse([p for p in root.iterdir() if p.name.startswith('.submission')])
            self.assertEqual(json.loads((root/'run/run.json').read_text())['status'],'passed')

    def test_expired_run_cannot_export_after_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build(ROOT,root/'agent');source=root/'agent/starter/train.py'
            spec=importlib.util.spec_from_file_location('expired_starter',source)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            run=root/'run';run.mkdir();schedule=source.parent/'schedule.json'
            for index in range(11):
                stage=run/f'stage-{index:02d}';stage.mkdir()
                (stage/'report.json').write_text('{"status":"passed"}')
            state={'seed':0,'schedule_sha256':module.hashlib.sha256(schedule.read_bytes()).hexdigest(),
                   'data':str((root/'data').resolve()),'status':'failed','started_unix':0}
            (run/'run.json').write_text(json.dumps(state))
            args=['train.py','--resume','--data',str(root/'data'),'--output',str(run)]
            with patch('sys.argv',args),patch.object(module.subprocess,'run') as execute:
                with self.assertRaisesRegex(RuntimeError,'Packaging time budget exhausted'):module.main()
                execute.assert_not_called()
            self.assertEqual(json.loads((run/'run.json').read_text())['status'],'failed')

if __name__=='__main__':unittest.main()
