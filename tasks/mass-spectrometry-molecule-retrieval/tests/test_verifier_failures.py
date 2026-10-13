"""Failures must not turn untrusted files into measured model rewards."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader
import main_verifier


class VerifierFailureTests(unittest.TestCase):
    def test_new_inputs_are_scored_without_a_false_reference_claim(self):
        metrics = {'hit_rate_at_1': 1.0}
        report = grader.reference_comparison(metrics, {'files': {'queries.jsonl': 'new'}})
        self.assertFalse(report['comparable'])
        self.assertIsNone(report['beats_reference'])
        frozen = grader.reference_comparison(metrics, {'files': grader.REFERENCE_INPUT_FILES})
        self.assertTrue(frozen['comparable'])
        self.assertAlmostEqual(frozen['reference_delta_hit_at_1'], 1 - grader.REFERENCE_HIT_RATE_AT_1)

    def test_trusted_outputs_remove_stale_rewards_and_deny_other_uid_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, sealed = root / 'logs', root / 'sealed/run'
            output.mkdir(mode=0o777)
            for name in ('reward.json', 'reward.txt'):
                (output / name).write_text('1')
            main_verifier.prepare_trusted_directories(output, sealed)
            self.assertFalse((output / 'reward.json').exists())
            self.assertFalse((output / 'reward.txt').exists())
            self.assertEqual(output.stat().st_mode & 0o022, 0)
            self.assertEqual(sealed.parent.stat().st_mode & 0o077, 0)

    def test_agent_failures_have_zero_reward_and_explicit_status(self):
        failures = [
            ('invalid_submission', ValueError('bad hash')),
            ('inference_timeout', subprocess.TimeoutExpired('infer.sh', 1)),
            ('inference_failed', subprocess.CalledProcessError(7, 'infer.sh')),
        ]
        for status, error in failures:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = argparse.Namespace(output=root / 'logs', sealed=root / 'sealed/run',
                                          input=root / 'input', submission=root / 'package',
                                          timeout=1, no_wait=True)
                with mock.patch.object(main_verifier, 'validate_input', return_value=({}, {'q': {}}, {})), \
                     mock.patch.object(main_verifier, 'validate_submission',
                                       side_effect=error if status == 'invalid_submission' else None), \
                     mock.patch.object(main_verifier, 'run_submission', side_effect=error):
                    main_verifier.evaluate(args)
                reward = json.loads((args.output / 'reward.json').read_text())
                report = json.loads((args.output / 'report.json').read_text())
                self.assertEqual(reward, {'reward': 0.0})
                self.assertEqual(report['metrics']['hit_rate_at_1'], 0)
                self.assertEqual(report['metrics']['mces_at_1'], 100)
                self.assertEqual(report['status'], status)
                self.assertFalse(report['model_quality_measured'])

    def test_setup_error_cannot_reuse_a_previous_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(output=root / 'logs', sealed=root / 'sealed/run',
                                      input=root / 'input', submission=root / 'package',
                                      timeout=1, no_wait=True)
            args.output.mkdir()
            (args.output / 'reward.txt').write_text('1')
            with mock.patch.object(main_verifier, 'validate_input', side_effect=ValueError('input hash mismatch')):
                with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
                    main_verifier.evaluate(args)
            self.assertFalse((args.output / 'reward.txt').exists())
            self.assertFalse((args.output / 'reward.json').exists())

    def test_non_string_query_ids_are_invalid_outputs_not_scorer_crashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rankings.jsonl'
            for query_id in ([], {}, None, 1):
                with self.subTest(query_id=query_id):
                    path.write_text(json.dumps({'query_id': query_id, 'ranking': ['a']}) + '\n')
                    rankings, error = grader.load_rankings(path, {'q': {'candidate_pool_id': 'p'}}, {'p': {'a': 'C'}})
                    self.assertEqual(rankings, {})
                    self.assertIsNotNone(error)

    def test_grader_error_is_reported_without_waiting_for_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(output=root / 'logs', sealed=root / 'sealed/run',
                                      input=root / 'input', submission=root / 'package',
                                      timeout=1, no_wait=False)
            def publish_error(*unused):
                result = root / 'sealed/results'
                result.mkdir()
                (result / 'error.json').write_text('{"error":"CBC unavailable"}')
            with mock.patch.object(main_verifier, 'validate_input', return_value=({}, {'q': {}}, {})), \
                 mock.patch.object(main_verifier, 'validate_submission'), \
                 mock.patch.object(main_verifier, 'run_submission', return_value=root / 'predictions'), \
                 mock.patch.object(main_verifier, 'seal_output', side_effect=publish_error):
                with self.assertRaisesRegex(RuntimeError, 'CBC unavailable'):
                    main_verifier.evaluate(args)
            self.assertFalse((args.output / 'reward.json').exists())
