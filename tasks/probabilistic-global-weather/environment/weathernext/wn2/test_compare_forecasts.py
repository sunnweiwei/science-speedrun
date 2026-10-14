import numpy as np
import numcodecs
import pytest
import zarr

from compare_forecasts import compare


def stores(tmp_path):
    paths = [tmp_path/'left', tmp_path/'right']
    for path, chunk, codec in zip(paths, (3, 2), (numcodecs.Zlib(), numcodecs.Blosc())):
        group = zarr.open_group(str(path), mode='w')
        array = group.create_dataset('wind', data=np.arange(7, dtype=np.float32),
                                     chunks=(chunk,), compressor=codec, fill_value=np.nan)
        array.attrs.update(_ARRAY_DIMENSIONS=['point'], units='m s-1')
        zarr.consolidate_metadata(str(path))
    return paths


def test_compression_and_chunk_layout_do_not_change_exact_values(tmp_path):
    assert compare(*stores(tmp_path))['status'] == 'passed'


def test_even_signed_zero_bit_change_fails(tmp_path):
    left, right = stores(tmp_path)
    zarr.open_group(str(right), mode='a')['wind'][0] = -0.0
    with pytest.raises(ValueError, match='bits differ'):
        compare(left, right)


def test_identically_missing_predictions_do_not_pass(tmp_path):
    paths = stores(tmp_path)
    for path in paths:
        zarr.open_group(str(path), mode='a')['wind'][0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        compare(*paths)


def test_units_mismatch_fails(tmp_path):
    left, right = stores(tmp_path)
    zarr.open_group(str(right), mode='a')['wind'].attrs['units'] = 'K'
    zarr.consolidate_metadata(str(right))
    with pytest.raises(ValueError, match='schema differs'):
        compare(left, right)
