"""Open directory stores and single-file Zarr ZipStores uniformly."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_STORED

import zarr


def is_zip_store(path: str | Path) -> bool:
    return str(path).endswith(".zarr.zip") or str(path).endswith(".zarr.partial.zip")


def open_group(path: str | Path, *, mode: str = "r"):
    path = Path(path)
    if is_zip_store(path):
        store = zarr.ZipStore(str(path), mode=mode, compression=ZIP_STORED, allowZip64=True)
        try:
            return zarr.open_group(store=store, mode=mode), store
        except BaseException:
            store.close()
            raise
    return zarr.open_group(str(path), mode=mode), None


def close_store(store) -> None:
    if store is not None:
        store.close()
