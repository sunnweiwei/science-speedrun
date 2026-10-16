"""The historical hour release retains its source identity and cost budgets."""
import copy
import json
from pathlib import Path
import sys
import unittest

DATA = Path(__file__).resolve().parents[1]/'environment/data'
sys.path.insert(0,str(DATA))
import hour_release as hour
import public_release as public


@unittest.skipUnless((DATA/'cost-evidence-hour-v1.json').exists(), 'Evidence not exported yet')
class HourReleaseTests(unittest.TestCase):
    def setUp(self):
        self.evidence=hour.read('cost-evidence-hour-v1.json')
        self.release=hour.build(self.evidence)

    def test_current_exact_membership_and_regime(self):
        archive = DATA/'release-hour-v1.json'
        self.assertEqual(public.load_release(archive) if archive.exists() else public.load_release(),self.release)
        self.assertEqual(self.release['case_count'],19)
        self.assertEqual(self.release['training']['case_ids'],self.release['test']['case_ids'])
        self.assertTrue(self.release['test']['one_shared_checkpoint'])
        self.assertFalse(self.release['test']['parameter_updates'])
        self.assertFalse(self.release['test']['normalizer_updates'])
        self.assertEqual(self.release['backend_counts'],{'jax':1,'jax-fluids':1,'maia':17})
        self.assertEqual(self.release['files']['bytes'],1_286_356_479)
        self.assertEqual(self.release['files']['count'],233)
        self.assertLess(self.release['files']['bytes'],500_000_000_000)

    def test_short_round_and_each_larger_phase_fit_explicit_budget(self):
        total=0.0
        for c in self.release['cases']:
            b=c['reference_training_budgets']
            self.assertEqual(b['single_case_research']['controls'],20000)
            self.assertLessEqual(b['single_case_research']['planning_seconds_with_25pct_reserve'],36000)
            total+=b['short_multitask_phase']['planning_seconds_with_25pct_reserve']
        self.assertLessEqual(total,36000)
        self.assertEqual(total,self.release['reference_training_budget']['short_round_serial_single_h100_planning_seconds'])
        self.assertEqual(self.release['reference_training_budget']['short_round_total_controls'],76000)
        self.assertFalse(self.release['reference_training_budget']['paper_convergence_claim'])
        self.assertFalse(self.release['reference_training_budget']['agent_training_limit'])

    def test_no_single_step_probe_can_replace_real_training(self):
        for field,value in [('training_controls',6),('ppo_epochs',0),('changed_parameter_tensors',0),
                            ('training_seconds',0),('status','failed')]:
            evidence=copy.deepcopy(self.evidence)
            evidence['cases'][0]['training'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                hour.build(evidence)

    def test_no_partial_or_nonfinite_episode(self):
        for field,value in [('return',float('nan')),('controls',2),('status','failed')]:
            evidence=copy.deepcopy(self.evidence)
            replay=evidence['cases'][0]['replay']
            replay[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                hour.build(evidence)

    def test_no_slow_or_extra_task_and_no_disguised_two_gpu_cost(self):
        for mutate in [lambda e:e['cases'][0].update(case='Cylinder_3D_Re3900'),
                       lambda e:e['cases'][0].update(hardware='2 H100'),
                       lambda e:e['cases'][0].update(previous_seconds_per_control=88.976),
                       lambda e:e['cases'].pop()]:
            evidence=copy.deepcopy(self.evidence)
            mutate(evidence)
            with self.assertRaises(ValueError):
                hour.build(evidence)

    def test_all_prior_manifests_preserved(self):
        for filename,digest in [
            ('release-coverage-v1.json','365e9531671ae4eb24491e880f7e6c1d9768013b5da2bb389df9342525ca1896'),
            ('release-speed-v1.json','23c53782f87b91c9be8babee3cab5e7cd16730c9dc79eb46d1b19c796be82890'),
            ('release-500gb-v1.json','226b7059b4b6b34bb4a8f169374f2a6bc74bc4f780b22ccbc3171c06b34778f3')]:
            self.assertEqual(public.digest(public.load_release(DATA/filename)),digest)

    def test_data_membership_does_not_claim_full_evaluator_certification(self):
        self.assertEqual(self.release['status'],'data-release-not-cfd-certified')
        self.assertEqual(self.release['test']['execution_manifest_status'],
                         'requires_native_protocol_binding_and_full_replay')
        self.assertFalse(self.release['test']['independent_hidden_split'])

    def test_runtime_repair_is_explicit_and_uncontrolled_failure_is_not_called_slow(self):
        adapter=self.release['source']['organizer_runtime_adapters']['Channel_3D_Retau180']
        self.assertEqual(adapter['profile'],'channel-72cube-local-actuator-clock-v1')
        cavity=next(c for c in self.release['exclusions'] if c.get('environment')=='Cavity_2D_Re7500')
        self.assertEqual(cavity['reason'],'nonfinite_native_uncontrolled_full_episode')
        self.assertNotIn('Cavity_2D_Re7500',self.release['training']['case_ids'])
        evidence=copy.deepcopy(self.evidence)
        evidence['channel_clock_check']['zero_final_field_max_abs_differences']['U']=1.0
        with self.assertRaisesRegex(ValueError,'invariance'):
            hour.build(evidence)


if __name__=='__main__':
    unittest.main()
