from __future__ import annotations

import numpy as np

from agbench.zarr_store import close_store, open_group


def test_zipstore_round_trip(tmp_path):
    path = tmp_path / "part-0000.zarr.zip"
    group, store = open_group(path, mode="w")
    group.create_dataset("values", data=np.arange(12), chunks=(4,))
    close_store(store)
    group, store = open_group(path, mode="r")
    assert np.array_equal(group["values"][:], np.arange(12))
    close_store(store)
