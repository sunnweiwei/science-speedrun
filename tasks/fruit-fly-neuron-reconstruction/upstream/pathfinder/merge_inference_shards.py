"""Merge complete shard sets; retain incomplete inference coverage explicitly."""
import argparse
import json
import re
from itertools import islice
from pathlib import Path
import sys
from infer_shape import canonical_edges, connected_motifs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external, sha256, write_json


def ordered_shards(rows, keys):
    if not rows:raise ValueError('No shards')
    n=rows[0].get('shards')
    if type(n)!=int or n<1 or len(rows)!=n:raise ValueError('Complete shard set required')
    ranks=[r.get('shard_index') for r in rows]
    if any(type(r)!=int for r in ranks) or sorted(ranks)!=list(range(n)):
        raise ValueError('Duplicate or missing shard rank')
    for r in rows:
        if r.get('ground_truth_used') is not False or r.get('paper_comparable') is not False:
            raise ValueError('Prediction-only independent implementation required')
        for key in ['shards',*keys]:
            if key not in r or r[key]!=rows[0][key]:raise ValueError('Shard provenance mismatch: '+key)
    return sorted(rows,key=lambda r:r['shard_index'])


def merge_sense(results, receipts):
    audits=ordered_shards(receipts,[])
    outputs=ordered_shards(results,[])
    return merge_sense_stream(zip(audits,outputs),len(audits))


def merge_sense_stream(pairs, count, require_complete=False, expected_eligible=None, expected_provenance=None):
    """Consume rank-ordered shards once; retain one shared eligible list.

    Accumulated processed records and edges still scale with the final output.
    This avoids retaining every shard's duplicate global coverage arrays.
    """
    if type(count)!=int or count<1:raise ValueError('Complete shard set required')
    keys=['segmentation_sha256','image_manifest_sha256','instance_index_sha256',
          'assembled_receipt_sha256','coverage_sha256','checkpoint_sha256','eligible_instance_ids',
          'skipped_geometric_context_instance_ids','total_indexed_instances','input_shape',
          'prompt_radius','shape_radius_nm','min_size','threshold','all_tiles_complete']
    if expected_eligible is not None and (type(expected_eligible)!=int or expected_eligible<0):
        raise ValueError('Invalid expected eligible count')
    if expected_provenance is not None and (not isinstance(expected_provenance,dict) or not expected_provenance or set(expected_provenance)-set(keys)):
        raise ValueError('Invalid expected SENSE provenance fields')
    processed=[];unprocessed=[];edges=[];seen=0
    for rank,(r,out) in enumerate(pairs):
        if rank>=count:raise ValueError('Too many shards')
        for row in (r,out):
            if (row.get('shards')!=count or row.get('shard_index')!=rank
                or row.get('ground_truth_used') is not False or row.get('paper_comparable') is not False):
                raise ValueError('Invalid ordered shard identity')
        if rank==0:
            base={k:r[k] for k in keys}
            first_output={k:out[k] for k in ('metadata','schema')}
            eligible=base['eligible_instance_ids']
            if expected_eligible is not None and len(eligible)!=expected_eligible:
                raise ValueError('Eligible count differs from admitted campaign')
            if any(a>=b for a,b in zip(eligible,islice(eligible,1,None))):raise ValueError('Invalid eligible IDs')
            excluded=base['skipped_geometric_context_instance_ids']
            if isinstance(excluded,dict):
                if (excluded.get('storage')!='ordered-excluded-id-digest/v1'
                    or type(excluded.get('count')) is not int
                    or excluded['count']!=base['total_indexed_instances']-len(eligible)
                    or excluded.get('encoding')!='ascending little-endian uint64'
                    or excluded.get('voxel_size_nm')!=[16]*3 or excluded.get('radius_nm')!=10000
                    or excluded.get('shape')!=first_output['metadata']['shape']
                    or not isinstance(excluded.get('sha256'),str)
                    or not re.fullmatch('[0-9a-f]{64}',excluded['sha256'])):
                    raise ValueError('Invalid compact geometric exclusion evidence')
        else:
            for key in keys:
                if key not in r or r[key]!=base[key]:raise ValueError('Shard provenance mismatch: '+key)
            for key in ('metadata','schema'):
                if out.get(key)!=first_output[key]:raise ValueError('Shard output provenance mismatch: '+key)
        for key,value in (expected_provenance or {}).items():
            if r.get(key)!=value:raise ValueError('SENSE input differs from admitted provenance: '+key)
        seen+=1
        assigned=eligible[r['shard_index']::r['shards']]
        done=[v['instance_id'] for v in r['processed']]
        if (r['assigned_instance_ids']!=assigned or done!=assigned[:len(done)]
            or r['unprocessed_instance_ids']!=assigned[len(done):] or r['calls']!=len(done)):
            raise ValueError('Shard candidate coverage mismatch')
        if require_complete and (r.get('budget_exhausted') is not False or r['unprocessed_instance_ids'] or len(done)!=len(assigned)):
            raise ValueError('Incomplete SENSE shard cannot enter full refinement')
        done_set=set(done)
        for edge in out['edges']:
            if edge.get('evidence_seed_id') not in done_set:raise ValueError('Candidate without processed seed')
        processed.extend(r['processed']);unprocessed.extend(r['unprocessed_instance_ids']);edges.extend(out['edges'])
    if seen!=count:raise ValueError('Complete shard set required')
    return dict(schema='pathfinder-candidate-edges/v1',edges=canonical_edges(edges),metadata=first_output['metadata'],
        ground_truth_used=False,paper_comparable=False,provenance={k:base[k] for k in keys if not k.endswith('instance_ids')},
        coverage=dict(eligible_instance_ids=eligible,processed=sorted(processed,key=lambda r:r['instance_id']),
            unprocessed_instance_ids=sorted(unprocessed),
            skipped_geometric_context_instance_ids=base['skipped_geometric_context_instance_ids'],
            processed_count=len(processed),eligible_count=len(eligible),
            eligible_instance_fraction=len(processed)/len(eligible) if eligible else None,
            total_instance_fraction=len(processed)/base['total_indexed_instances'] if base['total_indexed_instances'] else None,
            scope='geometric 10um interior representatives only; not full-volume coverage'))


def merge_shape(rows, candidates):
    keys=['segments_sha256','edges_sha256','checkpoint_sha256','voxel_size_nm','origin_nm',
          'axes','radius_um','points','motif_limit','truncated','allow_partial_context',
          'total_enumerated_motifs','index_sha256','coverage_sha256','max_nodes','seed']
    rows=ordered_shards(rows,keys);base=rows[0]
    all_motifs,truncated=connected_motifs(canonical_edges(candidates['edges']),base['max_nodes'],base['motif_limit'])
    if len(all_motifs)!=base['total_enumerated_motifs'] or truncated!=base['truncated']:
        raise ValueError('Global motif enumeration mismatch')
    motifs=[];unprocessed=[];scored={}
    for r in rows:
        assigned=all_motifs[r['shard_index']::r['shards']]
        actual=[tuple(v['nodes']) for v in r['motifs']]
        if actual!=assigned[:len(actual)]:raise ValueError('Motif assignment mismatch')
        unprocessed.extend(assigned[len(actual):]);motifs.extend(r['motifs'])
        for v in r['motifs']:
            if v['shape'] is not None:
                if not 0<=v['shape']<=1 or v.get('prediction_context_complete') is False:
                    raise ValueError('Invalid scored motif')
                if not v['context_complete'] and not base['allow_partial_context']:
                    raise ValueError('Unapproved partial context')
                if len(v['nodes'])==2:scored[tuple(v['nodes'])]=v
    edges=[]
    for e in canonical_edges(candidates['edges']):
        r=scored.get((e['source'],e['target']))
        if r is not None:edges.append(dict(e,shape=r['shape'],context_complete=r['context_complete']))
    return dict(**{k:base[k] for k in keys},schema='pathfinder-shape-merged/v1',edges=edges,
        motifs=sorted(motifs,key=lambda r:(len(r['nodes']),r['nodes'])),
        unprocessed_motifs=[list(v) for v in sorted(unprocessed)],
        full_context_scored=sum(v['shape'] is not None and v['context_complete'] for v in motifs),
        ground_truth_used=False,paper_comparable=False)


def write_merge_json(path, result):
    """Same JSON bytes as common.write_json without a whole serialized copy."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['sense','shape'],required=True)
    p.add_argument('--shards',nargs='+',type=Path,required=True)
    p.add_argument('--edges',type=Path,help='Merged SENSE input required for SHAPE')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stream-sense',action='store_true',help='Read SENSE shards one at a time; paths must be in numeric rank order')
    p.add_argument('--require-complete',action='store_true',help='Reject partial or budget-exhausted SENSE coverage; requires --stream-sense')
    p.add_argument('--expected-eligible',type=int,help='Independent admitted eligible-center count; requires --stream-sense')
    p.add_argument('--expected-provenance',type=Path,help='Trusted expected receipt fields, independent of shard outputs; requires --stream-sense')
    a=p.parse_args();out=external(a.output)
    if (a.require_complete or a.expected_eligible is not None or a.expected_provenance is not None) and (a.stage!='sense' or not a.stream_sense):
        raise ValueError('Strict coverage options require streamed SENSE merge')
    if out.exists():raise FileExistsError(out)
    sources=[]
    if a.stage=='sense':
        def pairs():
            for path in a.shards:
                receipt_path=path/'receipt.json';edge_path=path/'edges.json'
                receipt_hash=sha256(receipt_path)
                r=json.loads(receipt_path.read_text())
                if receipt_hash!=sha256(receipt_path):raise ValueError('Receipt changed during read')
                edge_hash=sha256(edge_path)
                if r.get('edges_sha256')!=edge_hash:raise ValueError('SENSE output hash mismatch')
                result=json.loads(edge_path.read_text())
                if edge_hash!=sha256(edge_path):raise ValueError('Edges changed during read')
                sources.append(dict(receipt_sha256=receipt_hash,edges_sha256=edge_hash))
                yield r,result
        # Caller supplies rank order; fail closed on shuffled or missing ranks.
        if a.stream_sense:
            result=merge_sense_stream(pairs(),len(a.shards),a.require_complete,a.expected_eligible,
                json.loads(a.expected_provenance.read_text()) if a.expected_provenance else None)
        else:
            loaded=list(pairs())
            result=merge_sense([out for r,out in loaded],[r for r,out in loaded])
    else:
        if not a.edges:raise ValueError('--edges required for SHAPE')
        rows=[json.loads(path.read_text()) for path in a.shards]
        if any(r.get('edges_sha256')!=sha256(a.edges) for r in rows):raise ValueError('SHAPE input hash mismatch')
        result=merge_shape(rows,json.loads(a.edges.read_text()))
        sources=[dict(shape_sha256=sha256(path)) for path in a.shards]
    result['source_shards']=sources
    if a.stage=='sense' and a.stream_sense:write_merge_json(out,result)
    else:write_json(out,result)


if __name__=='__main__':main()
