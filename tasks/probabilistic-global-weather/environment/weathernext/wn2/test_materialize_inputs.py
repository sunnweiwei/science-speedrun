import numpy as np
import pytest
from materialize_inputs import fill_sparse_context


def test_input_interpolation_uses_same_frame_level_and_wraps_longitude():
    x=np.zeros((2,3,4),dtype='float32');x[0]=10;x[1]=1000
    x[0,1,0]=np.nan
    out,indices=fill_sparse_context(x,[-90,0,90],minimum_valid_area_fraction=.8)
    assert out[0,1,0]==10 and indices==[[0,1,0]]
    np.testing.assert_array_equal(out[1],x[1])


def test_large_or_infinite_context_gaps_fail():
    with pytest.raises(ValueError,match='coverage'):fill_sparse_context(np.full((3,4),np.nan),[-90,0,90])
    x=np.zeros((3,4));x[0,0]=np.inf
    with pytest.raises(ValueError,match='Infinite'):fill_sparse_context(x,[-90,0,90])
