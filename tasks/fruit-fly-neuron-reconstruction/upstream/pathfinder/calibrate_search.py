"""Select a predeclared SHAPE search threshold on the frozen validation core.

Reuse full-volume model scores; reconcile globally before exporting the core.
No additional neural inference or core-local merging is performed.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from index_storage import load_index,TableRecords
from common import external,sha256,write_json
from reconcile import reconcile,indexed_nodes,remap_slabs
from prediction_core import core_slices
from evaluate_physical import score
from fetch_axonem import reserve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','assembled','scores','suite','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();root=external(a.root);out=external(a.output)
    if root not in out.parents or out.exists():raise ValueError('New output under budget root required')
    protocol=json.loads(a.protocol.read_text())
    if protocol['role']!='validation' or protocol['suite_sha256']!=sha256(a.suite/'suite.json'):
        raise ValueError('Only the predeclared validation reference is allowed')
    thresholds=protocol['search_thresholds']
    if thresholds!=[.2,.5,.8] or protocol['tie_break']!='largest_threshold':
        raise ValueError('Unexpected search plan; freeze a new protocol before inference')
    base_path=a.assembled/'segmentation.npy';index_path=a.assembled/'instance-index.json'
    base=np.load(base_path,mmap_mode='r',allow_pickle=False);base_hash=sha256(base_path)
    scores=json.loads(a.scores.read_text())
    if scores.get('segments_sha256')!=base_hash or scores.get('ground_truth_used') is not False or scores.get('allow_partial_context') is not False:
        raise ValueError('Require matching prediction-only full-volume scores')
    nodes=indexed_nodes(load_index(index_path),base.shape,base_hash)
    # Compute every global map before looking at reference labels or quality.
    searches=[reconcile(nodes,scores['edges'],[(r['nodes'],r['shape']) for r in scores['motifs']
        if r['shape'] is not None],t,include_singletons=False) for t in thresholds]
    sl,grid=core_slices(json.loads((a.assembled/'metadata.json').read_text()),json.loads((a.suite/'metadata.json').read_text()))
    crop=base[sl];reserve(root,len(thresholds)*crop.nbytes+10_000_000)
    out.mkdir(parents=True);rows=[]
    for threshold,search in zip(thresholds,searches):
        dest=out/str(threshold);dest.mkdir()
        arr=np.lib.format.open_memmap(dest/'segmentation.npy',mode='w+',dtype=crop.dtype,shape=crop.shape)
        remap_slabs(crop,arr,search['components']);arr.flush();del arr
        write_json(dest/'metadata.json',grid);write_json(dest/'global-search.json',search)
        metric=score(a.suite,dest,dest/'score')
        rows.append(dict(threshold=threshold,nerl=metric['nerl'],metrics_sha256=sha256(dest/'score/metrics.json'),
            global_search_sha256=sha256(dest/'global-search.json')))
    selected=max(rows,key=lambda r:(r['nerl'],r['threshold']))
    write_json(out/'selection.json',dict(schema='validation-search-selection/v1',selected=selected,candidates=rows,
        role='validation',suite_sha256=protocol['suite_sha256'],protocol_sha256=sha256(a.protocol),
        base_sha256=base_hash,scores_sha256=sha256(a.scores),index_sha256=sha256(index_path),
        adapter_sha256=sha256(__file__),test_used=False,
        rule='Maximum validation nERL; exact ties choose largest threshold. Same model scores and global search before core export.',
        next_step='Export a new frozen model package with selected search threshold before any test inference'))


if __name__=='__main__':main()
