"""Reject HDF5 storage that can resolve outside the sealed prediction file."""
from pathlib import Path
import math
import stat

import h5py


def validate_prediction_storage(path: Path, *, max_objects: int = 100_000,
                                max_logical_bytes: int = 64 * 1024 ** 3) -> None:
    """Inspect link/storage metadata before any AnnData matrix is materialized.

    HDF5 virtual datasets can return a harmless fill value while a source is absent,
    then read another file when opened in the trusted grader. A filesystem copy or
    checksum alone does not seal that dependency. Ordinary dense and CSR/CSC AnnData
    files contain none of the external-storage mechanisms rejected here.
    """
    path=Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('prediction must be a regular, non-symlink file')
    with h5py.File(path,'r') as handle:
        pending=[handle['/']];seen=set();count=0;logical_bytes=0
        while pending:
            group=pending.pop()
            address=h5py.h5o.get_info(group.id).addr
            if address in seen:continue
            seen.add(address)
            for name in group:
                count+=1
                if count>max_objects:raise ValueError('prediction HDF5 object limit exceeded')
                link=group.get(name,getlink=True)
                if not isinstance(link,h5py.HardLink):
                    raise ValueError('prediction contains an external or symbolic HDF5 link')
                obj=group[name]
                if isinstance(obj,h5py.Group):pending.append(obj)
                elif isinstance(obj,h5py.Dataset):
                    if obj.is_virtual:raise ValueError('prediction contains a virtual HDF5 dataset')
                    if obj.external:raise ValueError('prediction contains external raw HDF5 storage')
                    logical_bytes+=math.prod(obj.shape)*obj.dtype.itemsize
                    if logical_bytes>max_logical_bytes:
                        raise ValueError('prediction exceeds logical HDF5 allocation limit')
