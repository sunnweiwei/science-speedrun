"""Cost-aware membership tests; not native CFD/evaluator certification."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

DATA=Path(__file__).resolve().parents[1]/'environment/data'
sys.path.insert(0,str(DATA))
import public_release as public
import speed_release as speed


class SpeedReleaseTests(unittest.TestCase):
    def setUp(self):
        self.selection=speed.read('selection-speed-v1.json')
        self.evidence=speed.read('cost-evidence-speed-v1.json')
        archive=DATA/'release-speed-v1.json'
        self.release=public.load_release(archive) if archive.exists() else public.load_release()

    def test_reproducible_current_release_with_3d_and_exact_train_test(self):
        self.assertEqual(self.release,speed.build(self.selection,self.evidence))
        self.assertEqual(self.release['release_id'],speed.RELEASE_ID)
        ids=[c['id'] for c in self.release['cases']]
        self.assertEqual(ids,self.release['training']['case_ids'])
        self.assertEqual(ids,self.release['test']['case_ids'])
        self.assertTrue({'Channel_3D_Retau180','Pinball_3D_Re150','Sphere_3D_Re300'}.issubset(ids))
        self.assertGreaterEqual(sum(c['dimension']==3 for c in self.release['cases']),6)
        self.assertLessEqual(self.release['files']['bytes'],500_000_000_000)
        self.assertTrue(self.release['test']['one_shared_checkpoint'])
        self.assertFalse(self.release['test']['parameter_updates'])
        self.assertFalse(self.release['test']['normalizer_updates'])
        self.assertEqual(self.release['status'],'data-release-not-cfd-certified')

    def test_no_missing_slow_or_unmeasured_case_can_enter_silently(self):
        for name in ['Sphere_3D_Re3700','DRA2303Jet_3D_Re400000_Ma02','not-a-resource']:
            selection=copy.deepcopy(self.selection)
            selection['case_ids']=sorted(set(selection['case_ids'])|{name})
            with self.subTest(name=name),self.assertRaises(ValueError):
                speed.build(selection,self.evidence)

    def test_cost_source_and_budget_tampering_are_rejected(self):
        evidence=copy.deepcopy(self.evidence)
        evidence['cases'][0]['seconds_per_control']=0
        with self.assertRaisesRegex(ValueError,'evidence has changed'):
            speed.build(self.selection,evidence)
        selection=copy.deepcopy(self.selection)
        selection['max_source_bytes']=1_000_000_000_000
        with self.assertRaisesRegex(ValueError,'budget'):
            speed.build(selection,self.evidence)

    def test_incomplete_reset_and_sample_summary_are_rejected(self):
        name=self.selection['case_ids'][0]
        for field,value in [('reset_seconds',None),('container_exit_code',137),
                            ('timing_samples_seconds',[1])]:
            evidence=copy.deepcopy(self.evidence)
            next(r for r in evidence['cases'] if r['case']==name)[field]=value
            selection=copy.deepcopy(self.selection)
            selection['cost_evidence_sha256']=public.digest(evidence)
            with self.subTest(field=field),self.assertRaises(ValueError):
                speed.build(selection,evidence)

    def test_task_metadata_and_prompt_use_current_release(self):
        task=DATA.parents[1]
        config=tomllib.loads((task/'task.toml').read_text())
        current=public.load_release()
        self.assertEqual(config['metadata']['data_release'],current['release_id'])
        self.assertEqual(config['metadata']['data_budget_bytes'],500_000_000_000)
        self.assertIn(current['release_id'],(task/'instruction.md').read_text())

    def test_partial_suite_and_mutated_manifest_are_rejected(self):
        suite={'data_release_id':self.release['release_id'],
               'data_release_sha256':public.digest(self.release),
               'cases':[{k:c[k] for k in ['id','environment','backend']} for c in self.release['cases']]}
        public.validate_suite_binding(self.release,suite)
        suite['cases'].pop()
        with self.assertRaisesRegex(ValueError,'exact frozen'):
            public.validate_suite_binding(self.release,suite)
        with tempfile.TemporaryDirectory() as temporary:
            changed=copy.deepcopy(self.release)
            changed['test']['case_ids'].pop()
            path=Path(temporary)/'release.json'
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'exact frozen'):
                public.load_release(path)

    def test_default_materializer_plans_new_selection_and_legacy_remains_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination=Path(temporary)/'not-created'
            result=subprocess.run([sys.executable,str(DATA/'materialize.py'),str(destination)],
                                  capture_output=True,text=True,check=True)
            plan=json.loads(result.stdout)
            current=public.load_release()
            self.assertEqual(plan['directories'],current['test']['case_ids'])
            self.assertEqual(plan['bytes'],current['files']['bytes'])
            self.assertTrue(plan['complete_release_selection'])
            self.assertFalse(destination.exists())
        old=public.load_release(DATA/'release-500gb-v1.json')
        self.assertEqual(old['case_count'],47)
        self.assertEqual(old['files']['bytes'],488770756805)

    def test_verifier_does_not_accept_old_release_as_new_benchmark(self):
        sys.path.insert(0,str(DATA.parent/'evaluation'))
        import evaluate
        from protocol import InfrastructureError
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            old=public.load_release(DATA/'release-500gb-v1.json')
            (root/'release.json').write_text(json.dumps(old))
            (root/'suite.json').write_text(json.dumps({'data_release_id':old['release_id']}))
            with self.assertRaisesRegex(InfrastructureError,'verifier.*frozen selection'):
                evaluate.preflight(root,require_reference=False)


if __name__=='__main__':
    unittest.main()
