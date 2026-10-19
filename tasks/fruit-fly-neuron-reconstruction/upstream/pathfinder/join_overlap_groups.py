"""Join all four-GPU groups; partial groups never become a volume manifest."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from infer_overlap import merge


def collect(root,groups):
    ledger=json.loads((root/'budget.json').read_text())
    jobs={j['job_id']:j for j in ledger['jobs']}
    offsets=[];job_ids=[];all_rows=[];pins={};total=None;policy=None
    for folder in groups:
        folder=folder.resolve()
        if root not in folder.parents:raise ValueError('Groups must remain in original budget root')
        marker=folder/'group-complete.json';group=json.loads(marker.read_text())
        if (group.get('schema')!='overlap-worker-group/v1' or group.get('complete') is not True
            or group.get('ground_truth_used') is not False):raise ValueError('Complete prediction-only group required')
        n=group['total_shards'];offset=group['shard_offset']
        if type(n) is not int or n not in (8,12,16,20) or type(offset) is not int or offset%4 or not 0<=offset<=n-4:
            raise ValueError('Invalid global group geometry')
        if total is None:total=n;policy=group['assignment_policy']
        if n!=total or group['assignment_policy']!=policy:raise ValueError('Group scheduling policy differs')
        jid=group['job_id']
        if jid not in jobs or jobs[jid]['gpus']!=4:raise ValueError('Original registered four-GPU allocation required')
        offsets.append(offset);job_ids.append(jid)
        start_path=folder/'start.json';proof_path=folder/'workers-exited.json'
        if sha256(start_path)!=group['start_sha256'] or sha256(proof_path)!=group['workers_exited_sha256']:
            raise ValueError('Changed controller evidence')
        start=json.loads(start_path.read_text());proof=json.loads(proof_path.read_text())
        workers=proof['workers'];args=start['arguments']
        if (proof.get('schema')!='joined-ffn-workers/v1' or proof['job_id']!=jid or start['job_id']!=jid
            or len(workers)!=4 or [w['rank'] for w in workers]!=list(range(4))
            or len({w['pid'] for w in workers})!=4 or any(w['returncode']!=0 for w in workers)
            or args['total_shards']!=total or args['shard_offset']!=offset or args['assignment_policy']!=policy):
            raise ValueError('Four joined workers must match the declared group')
        expected={str(folder/f'shard-{rank}/receipt.json') for rank in range(offset,offset+4)}
        if set(group['shard_receipts'])!=expected:raise ValueError('Exactly four owned shard receipts required')
        for rank,w in enumerate(workers):
            command=w['command']
            for flag,value in (('--output',str(folder/f'shard-{offset+rank}')),('--shards',str(total)),
                               ('--shard-index',str(offset+rank)),('--assignment-policy',policy)):
                if command.count(flag)!=1 or command[command.index(flag)+1]!=value:
                    raise ValueError('Joined worker command differs from assigned global rank')
            path=folder/f'shard-{offset+rank}/receipt.json'
            if sha256(path)!=group['shard_receipts'][str(path)]:raise ValueError('Changed shard receipt')
            row=json.loads(path.read_text())
            if (row['job_id']!=jid or row['shard_index']!=offset+rank or row['shards']!=total
                or row['source_sha256']!=start['source_sha256']
                or row['checkpoint_sha256']!=start['input_sha256'][str(Path(args['checkpoint']).resolve())]
                or row['image_manifest_sha256']!=start['input_sha256'][str(Path(args['image_manifest']).resolve())]):
                raise ValueError('Shard provenance differs from group controller')
            all_rows.append(row);pins[str(path)]=sha256(path)
        pins.update({str(marker):sha256(marker),str(start_path):sha256(start_path),str(proof_path):sha256(proof_path)})
    if total is None or sorted(offsets)!=list(range(0,total,4)) or len(set(job_ids))!=len(job_ids):
        raise ValueError('All distinct four-GPU groups required exactly once')
    result=merge(all_rows)
    for path,digest in result['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Scientific source changed')
    result['shard_receipts']={p:h for p,h in pins.items() if p.endswith('/receipt.json')}
    result['group_evidence_sha256']=pins
    result['job_ids']=job_ids
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--groups',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();root=external(a.root);output=external(a.output)
    if root not in output.parents or output.exists():raise ValueError('New output inside original budget root required')
    result=collect(root,a.groups)
    from fetch_axonem import reserve
    reserve(root,10_000_000)
    write_json(output,result)


if __name__=='__main__':main()
