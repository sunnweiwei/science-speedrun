"""CPU tests of the scored verifier suite and the side-by-side GPU scheduler (no CFD)."""
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'environment/evaluation'))
from case_spec import VERIFIER_CASES, estimated_seconds, select_cases
from evaluate_native import SIDE_BY_SIDE_TIMEOUT_FACTOR, replay_cases


class VerifierSuiteTests(unittest.TestCase):
    def test_suite_keeps_every_family_dimension_and_backend(self):
        _, every = select_cases(['all'])
        _, scored = select_cases(['verifier'])
        self.assertEqual({c['id'] for c in scored}, set(VERIFIER_CASES))
        for key in ('family', 'backend'):
            self.assertEqual({c[key] for c in scored}, {c[key] for c in every}, key)
        self.assertEqual({'_2D' in c['id'] or c['id'] == 'Kolmogorov_2D' or c['id'].startswith('Nozzle2D')
                          for c in scored}, {True, False})
        twod = {c['id'] for c in every if '_3D' not in c['id']}
        self.assertTrue(twod <= set(VERIFIER_CASES), 'every 2D case is scored')
        self.assertTrue(any(c['solver_processes'] > 1 for c in scored), 'a two-GPU solver case is scored')

    def test_suite_fits_the_eight_gpu_budget(self):
        _, scored = select_cases(['verifier'])
        longest = max(estimated_seconds(c['id']) for c in scored)
        total = sum(estimated_seconds(c['id']) * c['solver_processes'] for c in scored)
        # Estimates are solo-H100 measurements; side by side the 8xH100 replay took 2 h 21 min (3.9 h on a shared host).
        self.assertLess(max(longest, total / 8), 2.5 * 3600)

    def test_scheduler_never_double_books_gpus(self):
        _, scored = select_cases(['verifier'])
        lock, busy, seen = threading.Lock(), set(), {}

        accounts = set()

        def fake(case, env):
            gpus = env['CUDA_VISIBLE_DEVICES'].split(',')
            uid = env['HYDROGYM_POLICY_UID']
            with lock:
                self.assertFalse(busy & set(gpus))
                self.assertNotIn(uid, accounts, 'two running cases share a policy account')
                busy.update(gpus)
                accounts.add(uid)
            self.assertEqual(len(gpus), case['solver_processes'])
            time.sleep(0.01)
            with lock:
                busy.difference_update(gpus)
                accounts.discard(uid)
            seen[case['id']] = gpus
            return {'status': 'ok'}

        out = replay_cases(scored, fake, [f'GPU-{i}' for i in range(8)])
        self.assertEqual(set(out), set(VERIFIER_CASES))

    def test_side_by_side_cases_get_a_longer_limit(self):
        _, scored = select_cases(['verifier'])
        limits = {}
        replay_cases(scored, lambda case, env: limits.setdefault(case['id'], case['timeout_sec']) and {'status': 'ok'},
                     [f'GPU-{i}' for i in range(8)])
        for case in scored:
            self.assertEqual(limits[case['id']], case['timeout_sec'] * SIDE_BY_SIDE_TIMEOUT_FACTOR)
        sequential = {}
        replay_cases(scored, lambda case, env: sequential.setdefault(case['id'], case['timeout_sec']) and {'status': 'ok'}, [])
        self.assertEqual(sequential, {c['id']: c['timeout_sec'] for c in scored})

    def test_scheduler_falls_back_to_sequential_without_gpus(self):
        _, scored = select_cases(['verifier'])
        out = replay_cases(scored, lambda case, env: {'status': 'ok', 'env': env}, [])
        self.assertTrue(all(row['env'] is None for row in out.values()))


if __name__ == '__main__':
    unittest.main()
