import unittest
import tempfile
import json
from pathlib import Path
from fetch_axonem import storage_limit,reserve
from unittest.mock import patch
from types import SimpleNamespace
import torch
from ffn_model import FFN15
from budget import allocation,within_budget,concurrency_allowed,settle_attempts

class BudgetRunTests(unittest.TestCase):
    def test_reservations_and_actual_charge(self):
        seconds,gpu=allocation([dict(reserved_seconds=100,gpus=4),dict(reserved_seconds=200,actual_seconds=30,gpus=2)])
        self.assertEqual((seconds,gpu),(130,460))

    def test_failed_requeue_keeps_both_attempt_charges(self):
        job=dict(job_id='123',reserved_seconds=2700,gpus=4)
        accounting='123|NODE_FAIL|2124\n123|FAILED|11\n123.batch|FAILED|11\n456|COMPLETED|99\n'
        settle_attempts(job,accounting)
        self.assertEqual(allocation([job]),(2135,8540))
        self.assertEqual(len(job['accounting_attempts']),2)
        settle_attempts(job,accounting)
        self.assertEqual(allocation([job]),(2135,8540))
        settle_attempts(job,'')
        self.assertEqual(allocation([job]),(2135,8540))

    def test_unfinished_retry_reserves_time_on_top_of_failed_attempt(self):
        job=dict(job_id='123',reserved_seconds=2700,gpus=4,actual_seconds=2124)
        # Row order must not turn an active retry into a settled failure.
        settle_attempts(job,'123|PENDING|0\n123|NODE_FAIL|2124\n')
        self.assertNotIn('actual_seconds',job)
        self.assertEqual(job['state'],'PENDING')
        self.assertEqual(allocation([job]),(4824,19296))
        self.assertFalse(concurrency_allowed([job],1,4))
        settle_attempts(job,'123|NODE_FAIL|2124\n123|RUNNING|300\n')
        self.assertEqual(allocation([job]),(4824,19296))
        settle_attempts(job,'123|NODE_FAIL|2124\n123|COMPLETED|600\n')
        self.assertEqual(allocation([job]),(2724,10896))
        self.assertNotIn('prior_attempt_seconds',job)
    def test_explicit_ledger_extension_preserves_storage_cap(self):
        ledger=dict(limits=dict(bytes=200000000000,allocation_seconds=86400,gpu_seconds=345600))
        self.assertFalse(within_budget(ledger,83000000000,114890,459560))
        ledger["limits"].update(allocation_seconds=115200,gpu_seconds=460800)
        self.assertTrue(within_budget(ledger,83000000000,114890,459560))
        self.assertFalse(within_budget(ledger,83000000000,115201,459560))
        self.assertFalse(within_budget(ledger,83000000000,114890,460801))
        self.assertFalse(within_budget(ledger,200000000001,114890,459560))

    def test_explicit_concurrency_counts_pending_reservations(self):
        jobs=[dict(gpus=4),dict(gpus=4,state='PENDING'),dict(gpus=4,actual_seconds=10)]
        self.assertFalse(concurrency_allowed(jobs,4))
        self.assertFalse(concurrency_allowed(jobs,4,8))
        self.assertTrue(concurrency_allowed(jobs,4,12))
        with self.assertRaises(ValueError):concurrency_allowed(jobs,4,0)

    def test_development_storage_does_not_extend_compute(self):
        ledger=dict(limits=dict(bytes=200000000000,allocation_seconds=100,gpu_seconds=400))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            self.assertFalse(within_budget(ledger,300000000000,100,400,root=root))
            policy=dict(schema=1,scope='development-only',root=str(root),
                        resident_limit_bytes=None,starter_target_bytes=200000000000,
                        user_authorization='Explicit test authorization')
            path=root/'development-storage-policy.json';path.write_text(json.dumps(policy))
            self.assertTrue(within_budget(ledger,300000000000,100,400,root=root))
            self.assertFalse(within_budget(ledger,300000000000,101,400,root=root))
            self.assertFalse(within_budget(ledger,300000000000,100,401,root=root))
            self.assertFalse(within_budget(ledger,300000000000,100,400))
            policy['root']=str(root/'different');path.write_text(json.dumps(policy))
            with self.assertRaises(ValueError):storage_limit(root)

    def test_reserve_skips_tree_walk_only_with_valid_development_waiver(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            with patch('fetch_axonem.usage',return_value=200000000000):
                with self.assertRaisesRegex(ValueError,'Starter storage'):reserve(root,1)
            (root/'development-storage-policy.json').write_text(json.dumps(dict(schema=1,scope='development-only',
                root=str(root),resident_limit_bytes=None,starter_target_bytes=200000000000,user_authorization='Explicit test authorization')))
            with patch('fetch_axonem.usage',side_effect=AssertionError('Unneeded tree scan')):
                with patch('fetch_axonem.shutil.disk_usage',return_value=SimpleNamespace(free=100)):
                    reserve(root,100)
                    with self.assertRaisesRegex(ValueError,'free space'):reserve(root,101)

    def test_submission_disk_scan_shortcut_requires_valid_root_waiver(self):
        from budget import observed_disk_usage
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger=dict(limits=dict(bytes=200000000000),bytes_on_disk=123)
            with patch('budget.usage',return_value=456) as scan:
                self.assertEqual(observed_disk_usage(root,ledger,True)[0],456);scan.assert_called_once()
            (root/'development-storage-policy.json').write_text(json.dumps(dict(schema=1,scope='development-only',root=str(root.resolve()),resident_limit_bytes=None,starter_target_bytes=200000000000,user_authorization='Explicit test authorization')))
            with patch('budget.usage',side_effect=AssertionError('Unneeded scan')):
                value,scope=observed_disk_usage(root,ledger,True)
                self.assertEqual(value,123);self.assertIn('not a current disk measurement',scope)
            (root/'development-storage-policy.json').write_text('{}')
            with self.assertRaises(ValueError):observed_disk_usage(root,ledger,True)

    def test_full_ffn_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        model=FFN15();self.assertEqual(len(model.blocks),20)
        raw=torch.rand(1,1,9,9,9);seed=torch.full_like(raw,-2.94)
        loss=model(raw,seed).square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()
