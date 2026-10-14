import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('wn2_materializer', Path(__file__).with_name('materialize.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MaterializerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = mock.Mock(target=self.root, wbx_source='/opt/weatherbenchX', image=None,
                              model_python=sys.executable, eval_python=sys.executable,
                              retries=1, job_timeout=5, verify_only=False)
        self.runner = m.Runner(self.args)

    def test_resume_rehashes_and_never_overwrites_corrupt_completion(self):
        data = self.root / 'data'
        data.write_bytes(b'original')
        marker = self.root / 'done.json'
        marker.write_text('{}')
        files = {'data': m.digest(data)}
        with mock.patch.object(self.runner, 'run') as run:
            result = self.runner.job('case', marker, lambda: m.checked_files(self.root, files), [])
            self.assertEqual(result[1], 'verified_existing')
            data.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Corrupt'):
                self.runner.job('case', marker, lambda: m.checked_files(self.root, files), [])
            run.assert_not_called()

    def test_retry_real_subprocess_then_resume_without_reexecution(self):
        counter = self.root / 'attempt'
        marker = self.root / 'done'
        script = ('from pathlib import Path; import sys; '
                  f'p=Path({str(counter)!r}); n=int(p.read_text())+1 if p.exists() else 1; '
                  'p.write_text(str(n)); '
                  f'Path({str(marker)!r}).write_text("ok") if n>1 else None; '
                  'sys.exit(0 if n>1 else 1)')
        with mock.patch.object(m.time, 'sleep'):
            self.runner.job('retry', marker, lambda: self.assertEqual(marker.read_text(), 'ok'), [sys.executable, '-c', script])
        self.runner.job('retry', marker, lambda: None, ['must-not-run'])
        self.assertEqual(counter.read_text(), '2')

    def test_offline_verification_does_not_prepare_missing_files(self):
        self.args.verify_only = True
        with mock.patch.object(self.runner, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Missing completed'):
                self.runner.job('missing', self.root / 'absent', lambda: None, [])
            run.assert_not_called()

    def test_timeout_is_bounded_and_logged(self):
        self.args.job_timeout = 0.05
        self.args.retries = 0
        with self.assertRaisesRegex(RuntimeError, 'failed; see'):
            self.runner.run('timeout', [sys.executable, '-c', 'import time; time.sleep(10)'])
        self.assertTrue((self.root / '.materialize/logs/timeout.log').exists())

    def test_inventory_cannot_escape_root(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            m.checked_files(self.root, {'../outside': 'fake'})

    def test_truth_inventory_cannot_omit_a_chunk(self):
        pp = self.root / 'protocol.json'
        p = dict(truth_source='HRES-fc0', variables={'temperature': {'pressure_levels': True}},
                 pressure_levels_hpa=[500, 850], lead_hours=[6])
        m.write(pp, p)
        case = self.root / 'cases/000.json'
        lock = {'sources': {'hres_fc0': dict(url='gs://fixture', metadata_sha256='metadata')}}
        m.write(case, dict(status='passed', case_index=0, protocol_sha256=m.digest(pp),
                           source='gs://fixture', metadata_sha256='metadata', chunks={'temperature/0.0.0.0.0': 'fake'}))
        with self.assertRaisesRegex(ValueError, 'Incomplete truth chunk inventory'):
            m.check_truth(self.root, pp, 0, lock)

    def test_protocol_mismatch_rejected_before_file_reads(self):
        pp = self.root / 'protocol.json'
        m.write(pp, {})
        m.write(self.root / 'inputs-000.json', dict(status='passed', case_index=0, protocol_sha256='changed'))
        with self.assertRaisesRegex(ValueError, 'Input protocol changed'):
            m.check_inputs(self.root, pp, 0, {})

    def test_pool_finishes_independent_work_but_fails_stage(self):
        completed = []
        def fail():
            raise RuntimeError('download failed')
        def success():
            completed.append(True)
            return 'other-case', 'prepared'
        with self.assertRaisesRegex(ValueError, 'download failed'):
            self.runner.pool([fail, success], 2)
        self.assertEqual(completed, [True])

    def test_training_dependencies_then_unified_evaluation(self):
        code = self.root / 'code'
        code.mkdir()
        m.write(code / 'training-protocol.json', dict(initializations=['2020-01-01T00:00:00'],
                offset_hours=[0, 6], calibration_end='2020-12-31T23:59:59'))
        for weather, precip, inputs, weather_dir, precip_dir in m.SPLITS.values():
            for filename, directory in ((weather, weather_dir), (precip, precip_dir)):
                pp = code / filename
                m.write(pp, dict(evaluation_initializations=['2021-01-01T00:00:00']))
                d = self.root / directory
                m.write(d / 'protocol.json', m.read(pp))
                m.write(d / 'cases/000.json', {})
                m.write(d / 'status.json', dict(status='passed', protocol_sha256=m.digest(pp),
                        case_manifests=[dict(file='cases/000.json', sha256=m.digest(d / 'cases/000.json'))]))
        self.args.scope = 'all'
        self.args.verify_only = True
        self.args.workers = self.args.compute_workers = 1
        completed = []
        def job(name, marker, check, command):
            if name.startswith('labels-'):
                self.assertIn('training-000', completed)
            if name == 'statistics-000':
                self.assertIn('labels-000-1', completed)
                self.assertIn('labels-000-.25', completed)
            if name == 'statistics-finalize':
                self.assertIn('statistics-000', completed)
            completed.append(name)
            return name, 'prepared'
        with mock.patch.object(m, 'CODE', code), mock.patch.object(m, 'check_truth_layout'), mock.patch.object(self.runner, 'job', side_effect=job), \
                mock.patch.object(self.runner, 'sources') as sources, mock.patch.object(self.runner, 'run') as run, \
                mock.patch.object(m, 'check_stats_final'):
            self.runner.execute()
            sources.assert_not_called()
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], 'unified-evaluation-resources')
            self.assertIn('--verify-only', run.call_args.args[1])
        self.assertEqual(len(completed), 8)

    def test_rolling_ibtracs_is_rejected_without_publishing_partial_file(self):
        self.args.scope = 'all'
        self.args.ibtracs = self.args.ibtracs_url = None
        self.runner.lock = {'sources': {}, 'ibtracs': {'url': 'https://example.invalid/snapshot', 'sha256': 'frozen'}}
        with mock.patch.object(m.urllib.request, 'urlopen', return_value=io.BytesIO(b'new revision')):
            with self.assertRaisesRegex(ValueError, 'IBTrACS differs'):
                self.runner.sources()
        self.assertFalse((self.root / 'IBTrACS.since1980.v04r01.nc').exists())


if __name__ == '__main__':
    unittest.main()
