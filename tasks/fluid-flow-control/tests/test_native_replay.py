"""Synthetic bookkeeping/adapter tests, never substitutes for native CFD runs."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
import case_spec
import evaluate_native as replay
from evaluate import package_hash
from protocol import InfrastructureError, PolicyError


class NativeProfilesTests(unittest.TestCase):
    def test_exact_frozen_coverage_and_semantics(self):
        spec, cases = case_spec.select_cases(['all'])
        release = case_spec.load_release()
        self.assertEqual([c['id'] for c in cases], [c['id'] for c in release['cases']])
        self.assertEqual(len(cases), 39)
        self.assertEqual(spec['data_release_sha256'], case_spec.digest(release))
        for c, source in zip(cases, release['cases']):
            self.assertEqual(c['runtime_profile'], source['runtime_profile'])
            self.assertEqual(c['solver_processes'], source['required_solver_gpus'])
            self.assertEqual(c['horizon'], source['selection_review']['native_episode_controls_for_costing'])
            self.assertEqual(c['seeds'], [123])
            self.assertEqual(c['episode_end'], 'native')
            self.assertGreaterEqual(c['timeout_sec'], 1800)
        self.assertEqual({c['backend'] for c in cases}, {'maia', 'jax', 'jax-fluids'})
        self.assertEqual(max(c['observation_size'] for c in cases), 128)
        self.assertEqual(max(c['action_size'] for c in cases), 24)

    def test_unknown_duplicate_and_mixed_all_rejected(self):
        for names in ([], ['unknown'], ['all', 'Cylinder_2D_Re200'],
                      ['Cylinder_2D_Re200', 'Cylinder_2D_Re200']):
            with self.subTest(names=names), self.assertRaises(InfrastructureError):
                case_spec.select_cases(names)

    def test_source_rejection_precedes_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises((InfrastructureError, OSError)):
                case_spec.verify_resources(Path(directory), case_spec.select_cases(['all'])[1])

    def test_missing_second_gpu_fails_before_native_solver_import(self):
        import run_native
        with tempfile.TemporaryDirectory() as directory:
            args = ['run_native.py', '--case', 'Cube_3D_Re3700', '--data', directory,
                    '--submission', directory, '--output', directory]
            with patch.object(sys, 'argv', args), patch.object(run_native, 'verify_resources'), \
                    patch.object(run_native, 'local_only'), \
                    patch.object(run_native.subprocess, 'check_output', return_value='["GPU-test"]'):
                with self.assertRaisesRegex(RuntimeError, 'requires 2 distinct GPUs'):
                    run_native.main()
            self.assertTrue((Path(directory) / 'launcher-error.json').exists())


class NativeReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.package = self.root / 'candidate'
        self.package.mkdir()
        (self.package / 'README.md').write_text('Synthetic bookkeeping fixture, not a model.\n')
        (self.package / 'infer.sh').write_text('exit 0\n')
        self.names = ['Cylinder_2D_Re200', 'Channel_3D_Retau180']
        self.output = self.root / 'results'
        self.verify = patch.object(replay, 'verify_resources').start()
        self.addCleanup(patch.stopall)
        self.snapshots = []

    def runner(self, case, data, package, output):
        self.assertNotEqual(package, self.package)
        self.assertEqual(package_hash(package), package_hash(self.package))
        self.snapshots.append(str(package))
        return {'status': 'ok', 'submission_sha256': package_hash(package), 'episodes': [
            {'seed': seed, 'steps': case['horizon'], 'return': -4.0,
             'primary_metric': dict(case['primary_metric'], status='ok', value=1.25),
             'terminated': True, 'truncated': False} for seed in case['seeds']]}

    def evaluate(self, runner=None):
        return replay.evaluate_native(self.data, self.package, self.output, self.names,
                                      runner=runner or self.runner)

    def test_relocated_same_package_no_anchors_or_suite_required(self):
        result = self.evaluate()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['successful_cases'], 2)
        self.assertFalse(result['benchmark_validated'])
        self.assertFalse(result['all_frozen_cases_selected'])
        self.assertEqual(len(set(self.snapshots)), 1)
        self.assertEqual(result['cases'][self.names[0]]['mean_return'], -4)
        self.assertEqual(result['cases'][self.names[0]]['primary_metric']['value'], 1.25)
        self.assertEqual(self.verify.call_count, 2)
        self.assertFalse((self.output / 'reward.json').exists())
        self.assertFalse((self.data / 'suite.json').exists())

    def test_all_cases_keep_one_package_and_full_accounting(self):
        self.names = ['all']
        result = self.evaluate()
        self.assertEqual(result['successful_cases'], 39)
        self.assertTrue(result['all_frozen_cases_selected'])
        self.assertFalse(result['benchmark_validated'])
        self.assertEqual(len(set(self.snapshots)), 1)

    def test_failures_are_explicit_and_remaining_cases_still_run(self):
        def runner(case, *args):
            if case['id'] == self.names[0]:
                raise InfrastructureError('Synthetic solver failure')
            row = self.runner(case, *args)
            row.update(status='policy_error', error='Synthetic malformed action')
            row.pop('episodes')
            return row
        result = self.evaluate(runner)
        self.assertEqual(result['status'], 'completed_with_errors')
        self.assertEqual(len(result['cases']), 2)
        self.assertEqual(result['cases'][self.names[0]]['status'], 'infrastructure_error')
        self.assertEqual(result['cases'][self.names[1]]['status'], 'policy_error')
        self.assertNotIn('mean_return', result['cases'][self.names[1]])
        self.assertFalse((self.output / 'reward.json').exists())

    def test_missing_package_records_every_requested_case(self):
        self.package = self.root / 'absent'
        result = self.evaluate()
        self.assertEqual(result['requested_cases'], 2)
        self.assertTrue(all(r['status'] == 'policy_error' for r in result['cases'].values()))

    def test_invalid_episode_or_package_identity_rejected(self):
        corruptions = [lambda row: row.update(submission_sha256='wrong'),
            lambda row: row['episodes'][0].update(steps=1),
            lambda row: row['episodes'][0].update(seed=0),
            lambda row: row['episodes'][0].update(**{'return': float('nan')}),
            lambda row: row['episodes'][0].pop('primary_metric'),
            lambda row: row['episodes'][0]['primary_metric'].update(value=float('nan')),
            lambda row: row['episodes'][0].update(terminated=False, truncated=False)]
        for i, corrupt in enumerate(corruptions):
            self.output = self.root / f'bad-{i}'
            def runner(*args):
                row = self.runner(*args)
                corrupt(row)
                return row
            with self.subTest(i=i):
                result = self.evaluate(runner)
                self.assertTrue(all(r['status'] == 'infrastructure_error' for r in result['cases'].values()))

    def test_snapshot_mutation_rejects_entire_run(self):
        def runner(case, data, package, output):
            row = self.runner(case, data, package, output)
            (package / 'README.md').write_text('Changed during replay')
            return row
        with self.assertRaisesRegex(InfrastructureError, 'snapshot mutated'):
            self.evaluate(runner)
        self.assertTrue((self.output / 'error.json').exists())
        self.assertFalse((self.output / 'results.json').exists())

    def test_resource_change_rejects_results(self):
        self.verify.side_effect = [None, InfrastructureError('Source changed')]
        with self.assertRaisesRegex(InfrastructureError, 'Source changed'):
            self.evaluate()
        self.assertFalse((self.output / 'results.json').exists())

    def test_code_change_rejects_results(self):
        with patch.object(replay, 'implementation_hash', side_effect=['before', 'after']):
            with self.assertRaisesRegex(InfrastructureError, 'implementation changed'):
                self.evaluate()

    def test_policy_session_cleanup_and_pid_reuse_protection(self):
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        ticks = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        receipt = self.root / 'policy-process.json'
        receipt.write_text(json.dumps({'pid': process.pid, 'start_ticks': 'stale'}))
        replay.stop_policy(self.root)
        self.assertIsNone(process.poll())
        receipt.write_text(json.dumps({'pid': process.pid, 'start_ticks': ticks}))
        replay.stop_policy(self.root)
        self.assertEqual(process.wait(timeout=3), -9)
        replay.stop_policy(self.root)

    def test_immediate_policy_exit_is_not_an_infrastructure_error(self):
        import native_envs
        import run_native
        env, policy = Mock(), Mock()
        policy.process.pid = 999999999
        case = case_spec.select_cases([self.names[0]])[1][0]
        with patch.object(native_envs, 'make_env', return_value=env), \
                patch.object(run_native, 'Policy', return_value=policy), \
                patch.object(run_native, 'replay_episode', side_effect=PolicyError('Exited')):
            run_native.worker(case, self.data, self.package, self.root)
        self.assertEqual(json.loads((self.root / 'result.json').read_text())['status'], 'policy_error')
        policy.abort.assert_called_once()
        env.close.assert_called_once()

    def test_overlapping_and_stale_outputs_rejected(self):
        for output in (self.data / 'output', self.package / 'output', self.root):
            self.output = output
            with self.subTest(output=output), self.assertRaises(InfrastructureError):
                self.evaluate()
        self.output = self.root / 'results'
        self.output.mkdir()
        (self.output / 'reward.json').write_text('{"score":1}')
        with self.assertRaisesRegex(InfrastructureError, 'empty'):
            self.evaluate()


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Native scalar adapters need NumPy from the task runtime')
class ReplayAdapterTests(unittest.TestCase):
    def test_native_completion_and_scalar_diagnostics(self):
        from native_envs import ReplayAdapter
        class Env:
            class Space:
                dtype, shape = np.float32, (1,)
            observation_space = action_space = Space()
            def reset(self, seed):
                self.count = 0
                return np.zeros(1), {}
            def step(self, action):
                self.count += 1
                return np.ones(1), np.float32(-1), self.count == 2, False, {'lift': np.array([2.])}
        with tempfile.TemporaryDirectory() as directory:
            env = ReplayAdapter(Env(), {'id': 'synthetic', 'horizon': 2}, Path(directory))
            env.reset(seed=123)
            self.assertFalse(env.step([0])[2])
            self.assertEqual(env.step([0])[4], {'lift': 2.0})
            self.assertEqual(json.loads((Path(directory) / 'progress.json').read_text())['phase'], 'complete')
            with self.assertRaises(InfrastructureError):
                env.step([0])
            env.case['horizon'] = 3
            env.reset(seed=123)
            env.step([0])
            with self.assertRaisesRegex(InfrastructureError, 'termination differs'):
                env.step([0])


if __name__ == '__main__':
    unittest.main()
