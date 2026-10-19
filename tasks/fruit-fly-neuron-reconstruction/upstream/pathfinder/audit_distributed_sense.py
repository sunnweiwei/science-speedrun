"""Verify distributed SENSE execution provenance before admitting its merged output.

This audit does not establish candidate coverage, numerical equivalence, SHAPE
completion or permission to adopt a full refinement phase. Those are separate
checks. It reads retained evidence only and never queries Slurm.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256,write_json


def audit(campaign_path, submissions_path, ledger_path, *, expected_campaign_sha256,
          expected_image, expected_context, expected_package_sha256):
    campaign_path=Path(campaign_path);base=campaign_path.parent
    submissions_path=Path(submissions_path)
    def read(path):return json.loads(Path(path).read_text())
    if sha256(campaign_path)!=expected_campaign_sha256:
        raise ValueError('Campaign identity differs')
    campaign=read(campaign_path);submissions_hash=sha256(submissions_path)
    if (campaign['image']!=str(expected_image) or campaign['context']!=str(expected_context)
        or campaign['package_manifest_sha256']!=expected_package_sha256):
        raise ValueError('Campaign model or data differs from caller contract')
    package=Path(campaign['model_package'])
    if sha256(package/'package.json')!=expected_package_sha256:
        raise ValueError('Package identity changed')
    n=campaign['shards'];eligible=campaign['eligible_centers']
    if type(n)!=int or n<=0 or type(eligible)!=int or eligible<0:
        raise ValueError('Invalid coverage cardinalities')
    counts=[len(range(rank,eligible,n)) for rank in range(n)]
    if campaign['rank_counts']!=counts:raise ValueError('Global rank sizes differ')
    groups=campaign['groups'];jobs=read(submissions_path)['jobs'];ledger=read(ledger_path)['jobs']
    if len(groups)!=len(jobs):raise ValueError('Missing submitted group')
    group_ids=[g['index'] for g in groups]
    if len(set(group_ids))!=len(groups) or set(group_ids)!={j['group'] for j in jobs}:
        raise ValueError('Missing or duplicate group identity')
    if len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Duplicate group job')
    evidence={};ranks=[]
    def pin(path,expected=None):
        path=Path(path);value=sha256(path)
        if expected is not None and value!=expected:raise ValueError('Evidence changed: '+str(path))
        evidence[str(path)]=value
    for p,h in campaign['source_pins'].items():pin(p,h)
    for group in groups:
        assigned=group['ranks'];gpus=group['gpus']
        if type(gpus)!=int or not 1<=gpus<=4 or len(assigned)!=gpus:
            raise ValueError('Invalid group GPU assignment')
        matches=[j for j in jobs if j['group']==group['index']]
        if len(matches)!=1 or matches[0]['ranks']!=assigned:raise ValueError('Submitted ranks differ')
        job=matches[0];registered=[j for j in ledger if j['job_id']==job['job_id']]
        if len(registered)!=1:raise ValueError('Missing unique ledger job')
        registered=registered[0];script=Path(group['script'])
        if (registered['gpus']!=gpus or registered['script']!=str(script)
            or registered['gpu_type']!=group['gpu_type']
            or registered['reserved_seconds']!=group['minutes']*60):
            raise ValueError('Ledger allocation differs from frozen group')
        pin(script,registered['script_sha256'])
        execution=base/f"group-{group['index']}/execution.json";pin(execution);row=read(execution)
        if (row['job_id']!=job['job_id'] or row['campaign_sha256']!=expected_campaign_sha256
            or row.get('complete') is not True or [r['rank'] for r in row['rows']]!=assigned):
            raise ValueError('Incomplete or mismatched group execution')
        for rank in row['rows']:
            if rank['returncode']!=0 or rank.get('complete') is not True:
                raise ValueError('Incomplete worker execution')
            pin(base/f"sense/shard-{rank['rank']}/receipt.json",rank['receipt_sha256'])
            ranks.append(rank['rank'])
    if sorted(ranks)!=list(range(n)):raise ValueError('Missing or overlapping global ranks')
    pin(campaign_path,expected_campaign_sha256);pin(submissions_path,submissions_hash)
    for p,h in evidence.items():
        if sha256(p)!=h:raise ValueError('Evidence changed during audit')
    return dict(schema='distributed-sense-execution-audit/v1',execution_provenance_valid=True,
                campaign_sha256=expected_campaign_sha256,shards=n,groups=len(groups),
                eligible_centers=eligible,evidence_sha256=evidence,
                candidate_coverage_verified=False,full_pipeline_complete=False,
                scope='Registered complete disjoint rank execution; strict merge, independent coverage and SHAPE still required')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','submissions','ledger','expected-image','expected-context','output'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('expected-campaign-sha256','expected-package-sha256'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous audit')
    result=audit(a.campaign,a.submissions,a.ledger,expected_campaign_sha256=a.expected_campaign_sha256,
                 expected_image=a.expected_image,expected_context=a.expected_context,
                 expected_package_sha256=a.expected_package_sha256)
    write_json(a.output,result)

if __name__=='__main__':main()
