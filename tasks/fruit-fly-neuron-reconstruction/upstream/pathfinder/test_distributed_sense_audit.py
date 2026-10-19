import json
from pathlib import Path
import tempfile
import unittest
from audit_distributed_sense import audit
from common import sha256

class DistributedSenseAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);r=self.root
        self.write('model/package.json',{})
        groups=[];jobs=[];ledger=[]
        for group,ranks in enumerate(([0,1],[2])):
            script=r/f'group-{group}.sbatch';script.write_text('#!/bin/bash\n')
            groups.append(dict(index=group,ranks=ranks,gpus=len(ranks),script=str(script),minutes=20,gpu_type='a5000'))
            jobs.append(dict(group=group,ranks=ranks,job_id=str(100+group)))
            ledger.append(dict(job_id=str(100+group),gpus=len(ranks),script=str(script),script_sha256=sha256(script),reserved_seconds=1200,gpu_type='a5000'))
            for rank in ranks:self.write(f'sense/shard-{rank}/receipt.json',dict(rank=rank))
        campaign=dict(shards=3,eligible_centers=8,rank_counts=[3,3,2],groups=groups,image='image',context='context',
                      package_manifest_sha256=sha256(r/'model/package.json'),model_package=str(r/'model'),source_pins={})
        self.write('campaign.json',campaign);self.write('submissions.json',dict(jobs=jobs));self.write('ledger.json',dict(jobs=ledger))
        self.kw=dict(expected_campaign_sha256=sha256(r/'campaign.json'),expected_image='image',expected_context='context',expected_package_sha256=campaign['package_manifest_sha256'])
        for g,j in zip(groups,jobs):
            self.write(f"group-{g['index']}/execution.json",dict(job_id=j['job_id'],campaign_sha256=self.kw['expected_campaign_sha256'],complete=True,
                rows=[dict(rank=i,returncode=0,complete=True,receipt_sha256=sha256(r/f'sense/shard-{i}/receipt.json')) for i in g['ranks']]))
    def write(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def run_audit(self):return audit(self.root/'campaign.json',self.root/'submissions.json',self.root/'ledger.json',**self.kw)
    def mutate(self,name,fn):
        d=json.loads((self.root/name).read_text());fn(d);self.write(name,d)
    def test_execution_is_not_full_result(self):
        result=self.run_audit();self.assertTrue(result['execution_provenance_valid'])
        self.assertFalse(result['candidate_coverage_verified']);self.assertFalse(result['full_pipeline_complete'])
    def test_missing_and_partial_groups(self):
        self.mutate('group-1/execution.json',lambda d:d.update(complete=False))
        with self.assertRaises(ValueError):self.run_audit()
        (self.root/'group-1/execution.json').unlink()
        with self.assertRaises(FileNotFoundError):self.run_audit()
    def test_wrong_ledger_resources(self):
        self.mutate('ledger.json',lambda d:d['jobs'][0].update(gpus=1))
        with self.assertRaises(ValueError):self.run_audit()
    def test_duplicate_submission(self):
        self.mutate('submissions.json',lambda d:d['jobs'].__setitem__(1,d['jobs'][0]))
        with self.assertRaises(ValueError):self.run_audit()
    def test_mixed_campaign(self):
        self.mutate('group-1/execution.json',lambda d:d.update(campaign_sha256='wrong'))
        with self.assertRaises(ValueError):self.run_audit()
    def test_changed_receipt_and_contract(self):
        self.kw['expected_image']='other-image'
        with self.assertRaises(ValueError):self.run_audit()
        self.kw['expected_image']='image';self.write('sense/shard-2/receipt.json',dict(changed=True))
        with self.assertRaises(ValueError):self.run_audit()
    def test_failed_worker(self):
        self.mutate('group-1/execution.json',lambda d:d['rows'][0].update(returncode=1))
        with self.assertRaises(ValueError):self.run_audit()

if __name__=='__main__':unittest.main()
