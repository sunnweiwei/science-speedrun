import hashlib
from pathlib import Path
import tempfile
import unittest

from profile_reuse import TIMING_SOURCES, invalidation_reasons


class ProfileReuseTest(unittest.TestCase):
    def test_orchestrator_change_reuses_but_compute_change_invalidates(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory)
            for name in (*TIMING_SOURCES, 'train.py'):
                (code/name).write_text('original')
            report = dict(status='passed', profile=True, stage={'name':'stage'},
                          source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in code.iterdir()})
            (code/'train.py').write_text('added independent seed argument')
            self.assertEqual(invalidation_reasons(report, report['stage'], code), [])
            (code/'native_model.py').write_text('different attention computation')
            self.assertEqual(invalidation_reasons(report, report['stage'], code),
                             ['computation source changed: native_model.py'])

    def test_missing_provenance_and_failed_report_do_not_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory)
            for name in TIMING_SOURCES:
                (code/name).write_text('original')
            reasons = invalidation_reasons(dict(status='failed', profile=True, stage={}), {}, code)
            self.assertIn('not a passed timing profile', reasons)
            self.assertEqual(len(reasons), 1+len(TIMING_SOURCES))


if __name__ == '__main__':
    unittest.main()
