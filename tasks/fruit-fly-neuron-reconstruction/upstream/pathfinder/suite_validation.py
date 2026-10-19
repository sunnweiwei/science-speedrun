"""Frozen reference validation shared by the single-checkpoint verifier."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import sha256
from prediction_contract import geometry

CODE = Path(__file__).resolve().parent


def verifier_matches(path,digest):
    """Allow only the audited reporting-text change; freeze scoring bytes.

    Suites frozen before 11aa759 retain their original hash. Reversing that
    commit's exact report-field edit must reconstruct every original byte.
    Neither the suite nor its scoring implementation is rewritten.
    """
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()==digest:return True
    if path.name!='verify_prediction.py':return False
    current=(b"model_execution_verified=False,reference_scope=manifest['scope'],\n"
             b"        scope='Prediction scored against the frozen reference; training provenance and transfer protocol are not established by this verifier')")
    original=b"model_execution_verified=False,scope=manifest['scope'])"
    if data.count(current)!=1:return False
    return hashlib.sha256(data.replace(current,original)).hexdigest()==digest


# Frozen suites pin common.py. The track rename (ai4sci-tasks#107) changed only the native Slurm
# partition name in its GPU guard, which the verifier container never executes; every scoring
# helper is byte-identical. Exactly that one renamed file is accepted for the old pin.
COMMON_RENAMED={'b0da879bfe1c80eebba4f352e62e80728a1051ae1ae163ec7b506e5559d5bcba':
                '006a839cac9d7d54eb11b63c20cbfaac559ac7a8fd218f5cc16ef1ed2cc7e701'}


def suite_pins(suite,expected_geometry):
    suite=Path(suite).resolve();manifest=json.loads((suite/'suite.json').read_text())
    if (manifest.get('schema')!='pathfinder-evaluation-suite/v1' or manifest.get('primary_metric')!='nerl'
        or manifest.get('boundary_margin_voxels')!=[0,0,0] or manifest.get('paper_comparable') is not False
        or set(manifest.get('files',{}))!={'metadata.json','skeletons.json','run_length.py'}
        or set(manifest.get('verifier_code',{}))!={'verify_prediction.py','evaluate_erl.py','prediction_contract.py','topology.py'}):
        raise ValueError('Frozen trusted zero-margin suite required')
    pins={str(suite/'suite.json'):sha256(suite/'suite.json')}
    for name,digest in manifest['files'].items():
        path=suite/name
        if path.is_symlink() or sha256(path)!=digest:raise ValueError('Frozen suite input changed')
        pins[str(path)]=digest
    for name,digest in manifest['verifier_code'].items():
        if not verifier_matches(CODE/name,digest):raise ValueError('Frozen verifier implementation differs: '+name)
    common=sha256(CODE.parent/'common.py')
    if common!=manifest['common_sha256'] and COMMON_RENAMED.get(manifest['common_sha256'])!=common:
        raise ValueError('Frozen common implementation differs')
    from evaluate_physical import EVALUATOR_SHA
    if pins[str(suite/'run_length.py')]!=EVALUATOR_SHA:raise ValueError('Pinned author evaluator required')
    meta=json.loads((suite/'metadata.json').read_text())
    from prediction_core import core_slices
    reference=geometry(meta,meta['shape'])
    if reference['voxel_size_nm']!=expected_geometry['voxel_size_nm']:
        raise ValueError('Reference and prediction geometry spacing differ')
    import numpy as np
    offset=(np.asarray(reference['origin_nm'])-expected_geometry['origin_nm'])/reference['voxel_size_nm']
    if not np.array_equal(offset,np.round(offset)):raise ValueError('Reference geometry is not grid aligned')
    try:core_slices(expected_geometry,reference)
    except ValueError as error:raise ValueError('Reference geometry is outside full prediction') from error
    return pins
