"""Read-only arithmetic cross-check, without importing task scoring adapters."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,json,math,os,resource,time
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def calculate(truth,labels):
    nodes=truth['nodes'];ids={n['id']:i for i,n in enumerate(nodes)}
    if truth['axes']!='zyx' or truth['coordinate_units']!='nm' or len(ids)!=len(nodes) or len(labels)!=len(nodes):raise ValueError('Reference/assignment geometry invalid')
    gt=[int(n['skeleton_id']) for n in nodes];pred=[int(x) for x in labels]
    if any(g<=0 for g in gt) or any(p<0 for p in pred):raise ValueError('Invalid labels')
    # The pinned funlib implementation first rounds node coordinates to float32.
    # Retain that convention, but accumulate edge groups with math.fsum.
    positions=np.array([n['position_nm'] for n in nodes],np.float32)
    membership=defaultdict(set);support=Counter(zip(gt,pred))
    for g,p in zip(gt,pred):membership[p].add(g)
    merged={p for p,groups in membership.items() if p and len(groups)>1}
    total_groups=defaultdict(list);correct_groups=defaultdict(list);seen=set()
    for u,v in truth['edges']:
        i,j=ids[u],ids[v];pair=tuple(sorted((u,v)))
        if u==v or pair in seen or gt[i]!=gt[j]:raise ValueError('Invalid reference edge')
        seen.add(pair);length=float(np.linalg.norm(positions[i]-positions[j]))
        total_groups[gt[i]].append(length)
        p=pred[i]
        if p and p==pred[j] and p not in merged:correct_groups[gt[i],p].append(length)
    lengths=[math.fsum(v) for v in total_groups.values()];total=math.fsum(lengths)
    denominator=math.fsum(x*x for x in lengths)
    numerator=math.fsum(math.fsum(v)**2 for v in correct_groups.values())
    if total<=0 or denominator<=0:raise ValueError('Positive reference length required')
    splits=defaultdict(set);merges=defaultdict(set)
    for (g,p),count in support.items():
        if p and count>=5:splits[g].add(p)
        if p and count>=10:merges[p].add(g)
    missing=sum(p==0 for p in pred)
    return dict(erl_nm=numerator/total,maximum_erl_nm=denominator/total,nerl=numerator/denominator,
        total_path_length_nm=total,split_errors=sum(len(v)-1 for v in splits.values()),
        merge_errors=sum(len(v)-1 for v in merges.values()),omitted_nodes=missing,omitted_node_fraction=missing/len(nodes))


def self_check(evaluator):
    import networkx as nx
    spec=importlib.util.spec_from_file_location('author_oracle',evaluator);oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    truth=dict(axes='zyx',coordinate_units='nm',nodes=[],edges=[])
    for g in range(4):
        for i in range(16):
            node=g*16+i;truth['nodes'].append(dict(id=node,skeleton_id=2**63+g+1,position_nm=[g*1234,i*16,(i%3)*8]))
            if i:truth['edges'].append([node-1,node])
    cases=[[n['skeleton_id'] for n in truth['nodes']],[0]*64,[123]*64]
    rng=np.random.default_rng(31)
    for _ in range(20):
        labels=[2**63+100+n['id']//8 for n in truth['nodes']]
        for index in rng.choice(64,8,replace=False):labels[index]=int(rng.choice([0,3,9]))
        cases.append(labels)
    for labels in cases:
        graph=nx.Graph();pm={p:i+1 for i,p in enumerate(sorted(set(labels)-{0}))};pm[0]=0
        for n in truth['nodes']:
            z,y,x=n['position_nm'];graph.add_node(n['id'],skeleton_id=n['id']//16+1,z=z,y=y,x=x)
        graph.add_edges_from(truth['edges']);lengths=oracle.get_skeleton_lengths(graph,['z','y','x'],'skeleton_id',store_edge_length='length')
        got=oracle.expected_run_length(graph,'skeleton_id','length',{i:pm[p] for i,p in enumerate(labels)},skeleton_lengths=lengths)
        expected=calculate(truth,labels)['erl_nm']
        if not math.isclose(got,expected,rel_tol=1e-6,abs_tol=1e-10):raise ValueError('Independent arithmetic disagrees with synthetic author oracle')
    if calculate(truth,cases[0])['nerl']!=1 or calculate(truth,cases[1])['nerl']!=0 or calculate(truth,cases[2])['nerl']!=0:raise ValueError('Perfect/omission/merge controls failed')
    return len(cases)


def audit(plan_path,output):
    plan_path=Path(plan_path);output=Path(output)
    if output.exists():raise ValueError('Audit output already exists')
    plan=json.loads(plan_path.read_text());begun=time.monotonic()
    if not 0<plan['relative_tolerance']<=1e-6 or not 0<plan['absolute_tolerance']<=1e-10:
        raise ValueError('Independent audit tolerances cannot be widened')
    required=set()
    for case in plan['cases']:
        required.update(str(Path(case['suite'])/name) for name in ('suite.json','metadata.json','skeletons.json','run_length.py'))
        required.update(str(Path(case['prediction'])/name) for name in ('metadata.json','segmentation.npy'))
        required.update(str(Path(case['scoring'])/name) for name in ('metrics.json','node-assignments.npz'))
    if not required.issubset(plan['input_sha256']):raise ValueError('Audit plan must pin every sampled and scored input')
    for path,digest in plan['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Pinned input changed: '+path)
    controls=self_check(Path(plan['cases'][0]['suite'])/'run_length.py');rows=[]
    for case in plan['cases']:
        suite=Path(case['suite']);prediction=Path(case['prediction']);scoring=Path(case['scoring'])
        frozen=json.loads((suite/'suite.json').read_text());truth=json.loads((suite/'skeletons.json').read_text())
        if frozen['boundary_margin_voxels']!=[0,0,0]:raise ValueError('Zero-margin suite required')
        for name,digest in frozen['files'].items():
            if sha(suite/name)!=digest:raise ValueError('Suite contents changed')
        grid=json.loads((prediction/'metadata.json').read_text());reported=json.loads((scoring/'metrics.json').read_text())
        if grid['axes']!='zyx' or grid['coordinate_convention']!='voxel_outer_corner':raise ValueError('ZYX outer-corner grid required')
        seg=np.load(prediction/'segmentation.npy',mmap_mode='r',allow_pickle=False)
        try:
            if list(seg.shape)!=grid['shape'] or seg.dtype.kind!='u':raise ValueError('Prediction grid mismatch')
            labels=[]
            for node in truth['nodes']:
                index=tuple(math.floor((p-o)/s) for p,o,s in zip(node['position_nm'],grid['origin_nm'],grid['voxel_size_nm']))
                if any(not 0<=i<n for i,n in zip(index,seg.shape)):raise ValueError('Reference node outside prediction')
                labels.append(int(seg[index]))
        finally:seg._mmap.close()
        with np.load(scoring/'node-assignments.npz',allow_pickle=False) as stored:
            if stored['node_id'].tolist()!=[n['id'] for n in truth['nodes']] or stored['prediction_id'].tolist()!=labels:raise ValueError('Independent physical sampling differs')
        values=calculate(truth,labels);differences={}
        for name,value in values.items():
            target=reported[name]
            if isinstance(value,int):
                if value!=target:raise ValueError('Count differs: '+name)
            elif not math.isclose(value,target,rel_tol=plan['relative_tolerance'],abs_tol=plan['absolute_tolerance']):raise ValueError('Arithmetic differs: '+name)
            differences[name]=float(value-target)
        rows.append(dict(domain=case['domain'],method=case['method'],sampled_nodes=len(labels),node_assignments_exact=True,
            recomputed=values,difference_from_report=differences,reported_metrics_sha256=sha(scoring/'metrics.json')))
        print(json.dumps(rows[-1]),flush=True)
        del truth,labels
    for path,digest in plan['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Input changed during arithmetic audit')
    result=dict(complete=True,job_id=os.environ.get('SLURM_JOB_ID'),plan_sha256=sha(plan_path),script_sha256=sha(Path(__file__)),
        controls=controls,rows=rows,seconds=time.monotonic()-begun,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        relative_tolerance=plan['relative_tolerance'],absolute_tolerance=plan['absolute_tolerance'],scope=plan['scope'])
    with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return result


def freeze_replay(replay,output):
    replay=Path(replay).resolve();output=Path(output)
    if output.exists():raise ValueError('Audit plan already exists')
    result=json.loads((replay/'model-replay.json').read_text())
    if result.get('schema')!='pathfinder-ensemble-replay-result/v1' or result.get('model_replayed') is not True:
        raise ValueError('Completed trusted full workflow required before freezing audit')
    plan=json.loads((replay/'replay-plan.json').read_text())
    if result['replay_plan_sha256']!=sha(replay/'replay-plan.json') or result['comparison_sha256']!=sha(replay/'score/comparison.json'):
        raise ValueError('Replay result binding changed')
    pred=replay/'prediction';suite=Path(plan['suite'])
    paths=[replay/'model-replay.json',replay/'replay-plan.json',replay/'score/comparison.json']
    for phase,digest in result['phases'].items():
        record=pred/(phase+'-phase.json')
        if sha(record)!=digest:raise ValueError('Phase evidence changed')
        paths.append(record)
    expected={f'member-{i}-{step}' for i in range(6) for step in ('infer','pack','retire')}
    expected.update(('consensus','context','decisions','baseline-0','baseline-1','baseline-2','baseline-merge','refine','finalize'))
    if set(result['phases'])!=expected:raise ValueError('All 27 named full ensemble phases required')
    cases=[]
    for method,folder in [('ffn',pred/'consensus'),('baseline',replay/'score/baseline-core'),('full',pred/'submission')]:
        scoring=replay/'score'/method
        if sha(scoring/'metrics.json')!=result['metric_sha256'][method]:raise ValueError('Metric binding changed')
        cases.append(dict(domain='frozen-full-test-core',method=method,suite=str(suite),prediction=str(folder),scoring=str(scoring)))
        paths.extend(suite/name for name in ('suite.json','metadata.json','skeletons.json','run_length.py'))
        paths.extend(folder/name for name in ('metadata.json','segmentation.npy'))
        paths.extend(scoring/name for name in ('metrics.json','node-assignments.npz'))
    frozen=dict(cases=cases,input_sha256={str(p):sha(p) for p in paths},relative_tolerance=1e-6,absolute_tolerance=1e-10,
                scope='Independent physical sampling and ERL/nERL/topology arithmetic for the completed full workflow; no paper-parity claim')
    with output.open('x') as f:f.write(json.dumps(frozen,indent=2)+'\n')
    return frozen


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    prepare=subs.add_parser('prepare');prepare.add_argument('--replay',type=Path,required=True);prepare.add_argument('--output',type=Path,required=True)
    verify=subs.add_parser('audit');verify.add_argument('--plan',type=Path,required=True);verify.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':freeze_replay(args.replay,args.output)
    else:audit(args.plan,args.output)


if __name__=='__main__':main()
