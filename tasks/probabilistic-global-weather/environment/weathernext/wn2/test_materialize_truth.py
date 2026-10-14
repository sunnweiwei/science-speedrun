import io
import json
import numpy as np
import pytest
import xarray as xr

import materialize_truth as module
from test_evaluate import fixture


def test_materialize_roundtrip_resume_and_corruption_gate(tmp_path,monkeypatch):
    _,truth,protocol=fixture()
    protocol_path=tmp_path/'protocol.json';protocol_path.write_text(json.dumps(protocol))
    output=tmp_path/'release'
    module.initialize(output,protocol_path)
    source=truth.temperature.isel(case=0,drop=True).rename(lead='time',lat='latitude',lon='longitude')
    source=source.drop_vars('valid_time').assign_coords(time=np.array(['2022-01-05T06','2022-01-05T12'],dtype='datetime64[ns]')).to_dataset()
    original=xr.open_zarr
    monkeypatch.setattr(module.xr,'open_zarr',lambda url,**kw: source.copy(deep=True) if str(url).startswith('gs://') else original(url,**kw))
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**k:io.BytesIO(b'fixed source metadata'))
    module.materialize(output,0)
    module.materialize(output,0)
    module.finalize(output)
    with original(output/'truth.zarr',chunks=None) as restored:
        np.testing.assert_array_equal(restored.temperature,truth.temperature)
        np.testing.assert_array_equal(restored.valid_time,truth.valid_time)
    assert json.loads((output/'status.json').read_text())['status']=='passed'
    record=json.loads((output/'cases/000.json').read_text())
    chunk=output/'truth.zarr'/next(iter(record['chunks']))
    chunk.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):module.finalize(output)


def test_refuse_overwrite_or_incomplete_release(tmp_path):
    _,_,protocol=fixture()
    path=tmp_path/'p.json';path.write_text(json.dumps(protocol))
    output=tmp_path/'release';module.initialize(output,path)
    with pytest.raises(ValueError,match='overwrite'):module.initialize(output,path)
    with pytest.raises(FileNotFoundError):module.finalize(output)
