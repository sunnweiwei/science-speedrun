"""Actual frozen release integration; short-loop evidence is not a policy score."""
import json
from pathlib import Path
import sys
import unittest

DATA=Path(__file__).resolve().parents[1]/'environment/data'
sys.path.insert(0,str(DATA))
import coverage_release
import public_release


@unittest.skipUnless((DATA/'selection-coverage-v1.json').exists(), 'Coverage decision not frozen yet')
class FrozenCoverageTests(unittest.TestCase):
    def test_current_manifest_is_exact_reviewed_build(self):
        archive=DATA/'release-coverage-v1.json'
        release=public_release.load_release(archive) if archive.exists() else public_release.load_release()
        self.assertEqual(release,coverage_release.build_current())
        self.assertLessEqual(release['files']['bytes'],500_000_000_000)
        self.assertGreater(release['case_count'],28)
        self.assertEqual(release['training']['case_ids'],release['test']['case_ids'])
        self.assertTrue(release['test']['one_shared_checkpoint'])
        self.assertFalse(release['test']['parameter_updates'])
        self.assertFalse(release['test']['normalizer_updates'])
        self.assertEqual(release['status'],'data-release-not-cfd-certified')
        for case in release['cases']:
            record=case['cost_screening']
            self.assertGreaterEqual(len(record['timing_samples_seconds']),2)
            self.assertEqual(record['container_exit_code'],0)
            self.assertEqual(case['execution_status'],'short_loop_passed_full_evaluation_pending')

    def test_previous_release_identities_are_preserved(self):
        for filename,expected in [
            ('release-speed-v1.json','23c53782f87b91c9be8babee3cab5e7cd16730c9dc79eb46d1b19c796be82890'),
            ('release-500gb-v1.json','226b7059b4b6b34bb4a8f169374f2a6bc74bc4f780b22ccbc3171c06b34778f3')]:
            with self.subTest(filename=filename):
                release=public_release.load_release(DATA/filename)
                self.assertEqual(public_release.digest(release),expected)

    def test_all_complete_candidates_have_one_decision(self):
        selection=json.loads((DATA/'selection-coverage-v1.json').read_text())
        included={r['case'] for r in selection['included']}
        excluded={r['case'] for r in selection['excluded_complete_resources']}
        self.assertFalse(included & excluded)
        self.assertEqual(len(included | excluded),62)
        self.assertTrue(selection['policy']['unknown_is_not_slow'])
        self.assertIsNone(selection['policy']['global_seconds_per_control_cutoff'])


if __name__=='__main__':
    unittest.main()
