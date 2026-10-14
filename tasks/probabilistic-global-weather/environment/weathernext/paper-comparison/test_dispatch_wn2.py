import json
import pytest
from dispatch_wn2 import digest,verify_final_package


def test_dispatch_requires_final_unmodified_package(tmp_path):
    assert verify_final_package(tmp_path) is None
    for name in ['weights.npz','native_model.py','predict.py']:(tmp_path/name).write_bytes(b'synthetic gate fixture')
    report=dict(status='passed',stage={'name':'hres_025deg_ar8'},weights_sha256=digest(tmp_path/'weights.npz'))
    (tmp_path/'training-report.json').write_text(json.dumps(report))
    manifest=dict(schema='wn2-starter-package/v1',profile_artifact=False,parameter_count=183785575,
        source_revision='89c4b2a77a1c57b328b909c575550fd2e5aadc9c',
        files={p.name:digest(p) for p in tmp_path.iterdir()})
    file=tmp_path/'package.json';file.write_text(json.dumps(manifest))
    assert verify_final_package(tmp_path)==digest(file)
    manifest['profile_artifact']=True;file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='final native'):verify_final_package(tmp_path)
    manifest['profile_artifact']=False;file.write_text(json.dumps(manifest))
    (tmp_path/'weights.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed package'):verify_final_package(tmp_path)


def test_dispatch_rejects_nonfinal_training_stage(tmp_path):
    for name in ['weights.npz','native_model.py','predict.py']:(tmp_path/name).write_bytes(b'synthetic gate fixture')
    report=dict(status='passed',stage={'name':'hres_025deg_ar7'},weights_sha256=digest(tmp_path/'weights.npz'))
    (tmp_path/'training-report.json').write_text(json.dumps(report))
    manifest=dict(schema='wn2-starter-package/v1',profile_artifact=False,parameter_count=183785575,
        source_revision='89c4b2a77a1c57b328b909c575550fd2e5aadc9c',files={p.name:digest(p) for p in tmp_path.iterdir()})
    (tmp_path/'package.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='AR8'):verify_final_package(tmp_path)
