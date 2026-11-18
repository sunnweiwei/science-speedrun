"""Storage references must not change meaning between inference and grading."""
from pathlib import Path
import sys

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prediction_safety import validate_prediction_storage


def normal_file(path, sparse_matrix=False):
    x=np.array([[1.,0.],[0.,2.]],dtype=np.float32)
    ad.AnnData(sparse.csr_matrix(x) if sparse_matrix else x,
               obs=pd.DataFrame(index=['a','b']),var=pd.DataFrame(index=['G1','G2'])).write_h5ad(path)


@pytest.mark.parametrize('sparse_matrix',[False,True])
def test_normal_anndata_passes(tmp_path,sparse_matrix):
    p=tmp_path/'prediction.h5ad';normal_file(p,sparse_matrix)
    validate_prediction_storage(p)


def test_external_link_refused_without_following_it(tmp_path):
    p=tmp_path/'prediction.h5ad';normal_file(p)
    with h5py.File(p,'a') as f:
        del f['X'];f['X']=h5py.ExternalLink(str(tmp_path/'absent-private.h5'),'X')
    with pytest.raises(ValueError,match='HDF5 link'):validate_prediction_storage(p)


def test_virtual_dataset_changes_after_source_appears_but_is_rejected(tmp_path):
    p=tmp_path/'prediction.h5ad';truth=tmp_path/'simulated-private.h5';normal_file(p)
    with h5py.File(p,'a',libver='latest') as f:
        del f['X'];layout=h5py.VirtualLayout(shape=(2,2),dtype='f4')
        layout[:,:]=h5py.VirtualSource(str(truth),'X',shape=(2,2))
        ds=f.create_virtual_dataset('X',layout,fillvalue=0)
        ds.attrs['encoding-type']='array';ds.attrs['encoding-version']='0.2.0'
    # The same sealed bytes first look like zeros, then expose a newly visible source.
    np.testing.assert_array_equal(ad.read_h5ad(p).X,np.zeros((2,2)))
    with pytest.raises(ValueError,match='virtual'):validate_prediction_storage(p)
    with h5py.File(truth,'w') as f:f.create_dataset('X',data=np.full((2,2),37,dtype='f4'))
    np.testing.assert_array_equal(ad.read_h5ad(p).X,np.full((2,2),37))
    with pytest.raises(ValueError,match='virtual'):validate_prediction_storage(p)


def test_external_raw_storage_refused(tmp_path):
    p=tmp_path/'prediction.h5ad';normal_file(p)
    with h5py.File(p,'a') as f:
        del f['X'];f.create_dataset('X',shape=(2,2),dtype='f4',external=[(str(tmp_path/'payload.bin'),0,h5py.h5f.UNLIMITED)])
    with pytest.raises(ValueError,match='external raw'):validate_prediction_storage(p)


def test_symbolic_link_and_filesystem_symlink_refused(tmp_path):
    p=tmp_path/'prediction.h5ad';normal_file(p)
    with h5py.File(p,'a') as f:f['loop']=h5py.SoftLink('/')
    with pytest.raises(ValueError,match='HDF5 link'):validate_prediction_storage(p)
    q=tmp_path/'alias.h5ad';q.symlink_to(p)
    with pytest.raises(ValueError,match='non-symlink'):validate_prediction_storage(q)


def test_development_and_verifier_safety_code_identical():
    root=Path(__file__).resolve().parents[1]
    assert (root/'tests/prediction_safety.py').read_bytes()==(root/'environment/evaluation/prediction_safety.py').read_bytes()


def test_tiny_sparse_storage_cannot_request_unbounded_dense_allocation(tmp_path):
    p=tmp_path/'allocation.h5ad';normal_file(p)
    with h5py.File(p,'a') as f:
        del f['X'];f.create_dataset('X',shape=(2**35,2),dtype='f4',chunks=(1,2),fillvalue=0.)
    assert p.stat().st_size<1_000_000
    with pytest.raises(ValueError,match='allocation limit'):validate_prediction_storage(p)
