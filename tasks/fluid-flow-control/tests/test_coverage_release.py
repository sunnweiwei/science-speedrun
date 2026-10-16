"""Synthetic selection-validator fixtures, never scientific measurement evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

DATA=Path(__file__).resolve().parents[1]/'environment/data'
sys.path.insert(0,str(DATA))
import public_release as public
import coverage_release as coverage


class CoverageReleaseTests(unittest.TestCase):
    def setUp(self):
        previous=json.loads((DATA/'selection-speed-v1.json').read_text())
        self.evidence=json.loads((DATA/'cost-evidence-speed-v1.json').read_text())
        extra='Cube_3D_Re300'
        record=next(r for r in self.evidence['cases'] if r['case']==extra)
        # Explicit synthetic samples exercise >30 s acceptance. Do not export
        # them, or confuse validator tests with completed additional CFD probes.
        record.update(status='complete',container_exit_code=0,reset_seconds=50,
            timing_samples_seconds=[40,44,42],seconds_per_control=42,measured_controls=3)
        ids=sorted(previous['case_ids']+[extra])
        self.selection={'release_id':coverage.RELEASE_ID,'status':'frozen',
            'created':'test-fixture','max_source_bytes':500_000_000_000,
            'candidate_inventory_count':62,'cost_evidence_sha256':public.digest(self.evidence),
            'included':[{'case':n,'scientific_value':'Unit-test fixture only',
                'cost_review':'Artificial review for validator coverage',
                'runtime_profile':'synthetic-test-only'} for n in ids],
            'excluded_complete_resources':[e for e in previous['excluded_complete_resources'] if e['case']!=extra],
            'policy':{'global_seconds_per_control_cutoff':None}}

    def test_reviewed_slower_case_can_expand_first_set_under_budget(self):
        result=coverage.build(self.selection,self.evidence)
        self.assertEqual(result['case_count'],29)
        self.assertLess(result['files']['bytes'],500_000_000_000)
        self.assertEqual(result['training']['case_ids'],result['test']['case_ids'])
        self.assertTrue(result['test']['one_shared_checkpoint'])
        self.assertFalse(result['test']['parameter_updates'])
        self.assertEqual(result['status'],'data-release-not-cfd-certified')

    def test_missing_value_cost_or_profile_review_is_rejected(self):
        for field in ['scientific_value','cost_review','runtime_profile']:
            selection=copy.deepcopy(self.selection)
            del selection['included'][0][field]
            with self.subTest(field=field),self.assertRaises(ValueError):
                coverage.build(selection,self.evidence)

    def test_old_hard_cutoff_and_larger_budget_are_rejected(self):
        selection=copy.deepcopy(self.selection)
        selection['policy']['global_seconds_per_control_cutoff']=30
        with self.assertRaisesRegex(ValueError,'universal'):
            coverage.build(selection,self.evidence)
        selection=copy.deepcopy(self.selection)
        selection['max_source_bytes']=501_000_000_000
        with self.assertRaisesRegex(ValueError,'budget'):
            coverage.build(selection,self.evidence)

    def test_actual_source_sum_cannot_exceed_cap(self):
        # All timing values here are synthetic validator fixtures only.
        evidence=copy.deepcopy(self.evidence)
        for record in evidence['cases']:
            record.update(status='complete',container_exit_code=0,reset_seconds=1,
                timing_samples_seconds=[1,1],measured_controls=2,seconds_per_control=1)
        selection=copy.deepcopy(self.selection)
        selection['included']=[{'case':r['case'],'scientific_value':'synthetic',
            'cost_review':'synthetic','runtime_profile':'synthetic'}
            for r in sorted(evidence['cases'],key=lambda r:r['case'])]
        selection['excluded_complete_resources']=[]
        selection['cost_evidence_sha256']=public.digest(evidence)
        with self.assertRaisesRegex(ValueError,'budget|500'):
            coverage.build(selection,evidence)

    def test_incomplete_evidence_cannot_become_an_inclusion(self):
        evidence=copy.deepcopy(self.evidence)
        next(r for r in evidence['cases'] if r['case']=='Cube_3D_Re300')['status']='timeout'
        selection=copy.deepcopy(self.selection)
        selection['cost_evidence_sha256']=public.digest(evidence)
        with self.assertRaisesRegex(ValueError,'clean'):
            coverage.build(selection,evidence)

    def test_zero_field_response_requires_correction_before_selection(self):
        evidence=copy.deepcopy(self.evidence)
        next(r for r in evidence['cases'] if r['case']=='Cube_3D_Re300')['observed_field_response']=False
        selection=copy.deepcopy(self.selection)
        selection['cost_evidence_sha256']=public.digest(evidence)
        with self.assertRaisesRegex(ValueError,'zero-field-response'):
            coverage.build(selection,evidence)

    def test_two_gpu_allocation_without_distinct_rank_binding_is_rejected(self):
        evidence=copy.deepcopy(self.evidence)
        record=next(r for r in evidence['cases'] if r['case']=='Cube_3D_Re300')
        record.update(hardware='2 H100',gpu_bindings=[
            {'visible_cuda_devices':1,'cuda_device_uuid':'same-GPU'},
            {'visible_cuda_devices':1,'cuda_device_uuid':'same-GPU'}])
        selection=copy.deepcopy(self.selection)
        selection['cost_evidence_sha256']=public.digest(evidence)
        with self.assertRaisesRegex(ValueError,'CUDA rank bindings'):
            coverage.build(selection,evidence)

    def test_existing_cases_and_exclusion_accounting_cannot_disappear(self):
        selection=copy.deepcopy(self.selection)
        selection['included'].pop(0)
        with self.assertRaisesRegex(ValueError,'preserve'):
            coverage.build(selection,self.evidence)
        selection=copy.deepcopy(self.selection)
        selection['excluded_complete_resources'].pop()
        with self.assertRaisesRegex(ValueError,'accounting'):
            coverage.build(selection,self.evidence)

    def test_exact_coverage_loader_and_tampering_rejection(self):
        expected=coverage.build(self.selection,self.evidence)
        with tempfile.TemporaryDirectory() as temporary,patch.object(coverage,'build_current',return_value=expected):
            path=Path(temporary)/'release.json'
            path.write_text(json.dumps(expected))
            self.assertEqual(public.load_release(path),expected)
            changed=copy.deepcopy(expected)
            changed['test']['case_ids'].pop()
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'exact frozen'):
                public.load_release(path)


if __name__=='__main__':
    unittest.main()
