"""Freeze eight spatial octants of a 512-cube reference without new skeletons.

Octants are correlated descriptive slices; the full-core result is primary.
No predictions or model outputs are read, and no octant is selected by score.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256,write_json
from evaluate_physical import reference_bounds,EVALUATOR_SHA


def partition(source, output):
    source=Path(source);output=external(output)
    frozen=json.loads((source/'suite.json').read_text())
    for name,digest in frozen['files'].items():
        if sha256(source/name)!=digest:raise ValueError('Reference integrity mismatch')
    if sha256(source/'run_length.py')!=EVALUATOR_SHA:raise ValueError('Evaluator mismatch')
    meta=json.loads((source/'metadata.json').read_text());truth=json.loads((source/'skeletons.json').read_text())
    low,high=reference_bounds(meta);spacing=np.asarray(meta['voxel_size_nm'])
    if meta['axes']!='zyx' or not np.array_equal((high-low)/spacing,[512]*3):
        raise ValueError('Frozen 512-cube ZYX reference required')
    positions=np.asarray([n['position_nm'] for n in truth['nodes']])
    if (positions<low).any() or (positions>=high).any():raise ValueError('Reference outside core')
    output.mkdir(parents=True,exist_ok=False);records=[];assigned=set()
    for bits in itertools.product((0,1),repeat=3):
        lo=low+np.asarray(bits)*256*spacing;hi=lo+256*spacing
        keep=((positions>=lo)&(positions<hi)).all(axis=1)
        nodes=[n for n,k in zip(truth['nodes'],keep) if k];ids={n['id'] for n in nodes}
        if assigned&ids:raise ValueError('Octant node overlap')
        assigned|=ids
        edges=[e for e in truth['edges'] if e[0] in ids and e[1] in ids]
        name='octant-'+''.join(map(str,bits));folder=output/name;folder.mkdir()
        write_json(folder/'skeletons.json',dict(truth,nodes=nodes,edges=edges))
        write_json(folder/'metadata.json',dict(meta,scored_core_origin_nm=lo.tolist(),scored_core_shape=[256]*3))
        shutil.copyfile(source/'run_length.py',folder/'run_length.py')
        child=dict(frozen,files={n:sha256(folder/n) for n in ('metadata.json','skeletons.json','run_length.py')},
            scope='Fixed octant; correlated descriptive result, full 512-cube is primary',
            lineage=dict(parent_suite_sha256=sha256(source/'suite.json'),octant_zyx=list(bits),
                         adapter_sha256=sha256(__file__),policy='Induced subgraph; retain original IDs/coordinates'))
        write_json(folder/'suite.json',child)
        records.append(dict(region=name,nodes=len(nodes),edges=len(edges),evaluable=bool(edges),suite_sha256=sha256(folder/'suite.json')))
    if len(assigned)!=len(truth['nodes']):raise ValueError('Partition lost reference nodes')
    result=dict(parent_suite_sha256=sha256(source/'suite.json'),regions=records,independent_samples=False,
        primary_result='Unpartitioned 512-cube',cross_octant_edges_omitted=len(truth['edges'])-sum(r['edges'] for r in records))
    write_json(output/'partition.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(partition(a.source,a.output),indent=2))
