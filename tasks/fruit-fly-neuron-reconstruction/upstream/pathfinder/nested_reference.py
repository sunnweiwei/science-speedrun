"""Derive a centered scoring region without reskeletonizing or reading predictions."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate_physical import EVALUATOR_SHA, reference_bounds, reference_graph
from common import external, sha256, write_json


def subset(truth, metadata, core_voxels):
    if metadata.get('axes') != 'zyx' or type(core_voxels) != int or core_voxels <= 0:
        raise ValueError('Positive integer size and ZYX reference required')
    reference_graph(truth)
    low, high = reference_bounds(metadata)
    spacing = np.asarray(metadata['voxel_size_nm'], dtype=float)
    source_shape = (high-low)/spacing
    if (source_shape != np.round(source_shape)).any() or (source_shape < core_voxels).any():
        raise ValueError('Requested core exceeds integer reference grid')
    offset = (source_shape-core_voxels)/2
    if (offset != np.round(offset)).any():
        raise ValueError('Centered core must align with reference voxels')
    inner_low = low+offset*spacing
    inner_high = inner_low+core_voxels*spacing
    nodes = []
    for node in truth['nodes']:
        p = np.asarray(node['position_nm'])
        if (p < low).any() or (p >= high).any():
            raise ValueError('Source reference contains out-of-region nodes')
        if (p >= inner_low).all() and (p < inner_high).all():
            nodes.append(node)
    ids = {n['id'] for n in nodes}
    edges = [e for e in truth['edges'] if e[0] in ids and e[1] in ids]
    if not edges:
        raise ValueError('Preselected nested region has no reference edges; report it as invalid')
    result = dict(truth, nodes=nodes, edges=edges)
    meta = dict(metadata, scored_core_origin_nm=inner_low.tolist(), scored_core_shape=[core_voxels]*3)
    return result, meta


def derive(source, output, core_voxels):
    source = Path(source); output = external(output)
    frozen = json.loads((source/'suite.json').read_text())
    names = ('metadata.json', 'skeletons.json', 'run_length.py')
    if (frozen.get('schema') != 'pathfinder-evaluation-suite/v1' or
            frozen.get('boundary_margin_voxels') != [0,0,0] or set(frozen['files']) != set(names)):
        raise ValueError('Frozen zero-margin reference suite required')
    for name in names:
        if sha256(source/name) != frozen['files'][name]:
            raise ValueError('Source integrity mismatch: '+name)
    if sha256(source/'run_length.py') != EVALUATOR_SHA:
        raise ValueError('Pinned evaluator required')
    truth = json.loads((source/'skeletons.json').read_text())
    metadata = json.loads((source/'metadata.json').read_text())
    cropped, meta = subset(truth, metadata, core_voxels)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'skeletons.json', cropped)
    write_json(output/'metadata.json', meta)
    shutil.copyfile(source/'run_length.py', output/'run_length.py')
    lineage = dict(parent_suite_sha256=sha256(source/'suite.json'),
        adapter_sha256=sha256(Path(__file__)), source_nodes=len(truth['nodes']),
        retained_nodes=len(cropped['nodes']), source_edges=len(truth['edges']),
        retained_edges=len(cropped['edges']), policy='Centered induced subgraph; no reskeletonization; preserve IDs and physical positions')
    frozen.update(files={n:sha256(output/n) for n in names}, lineage=lineage,
        scope='Nested physical scoring region from same parent reference; correlated scale comparison', paper_comparable=False)
    write_json(output/'suite.json', frozen)
    return frozen


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--core-voxels', type=int, required=True)
    a = p.parse_args()
    print(json.dumps(derive(a.source, a.output, a.core_voxels), indent=2))
