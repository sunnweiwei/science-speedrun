"""User-dir wrapper around the official Uni-Mol2 pretraining code (Uni-Mol 90f52c4 unimol2/); no official file is edited.

The official LMDBDataset loads a different shard every epoch but keeps the epoch-1 batch sampler, so it assumes that
all shards have the same length (the upstream shards all hold 10,485,760 molecules). The leakage filter removed a
few molecules from each shard, and shard 11 lost the 100k-molecule validation hold-out, so a later shard can be
shorter than the first and the sampler then asks for an index past its end (IndexError on one rank, which Uni-Core's
final barrier turns into a silent hang). Every training shard is therefore truncated to one common length
(UNIMOL2X_SHARD_LEN, default 10,485,658 = the shortest of the rotated shards 0-10); shard 11 is left out of the
rotation (pretrain.sh links epoch e to /workspace/data/pretrain/train_part_{e % 11}.lmdb).
"""
import os
import sys

sys.path.insert(0, "/opt/Uni-Mol/unimol2")
import unimol2  # noqa: E402,F401  (registers the official tasks, models and losses)
from unimol2.data import lmdb_dataset as _lmdb  # noqa: E402

SHARD_LEN = int(os.environ.get("UNIMOL2X_SHARD_LEN", "10485658"))
_load = _lmdb.LMDBDataset.load_data


def _load_equal_length(self, data_dir, epoch, split):
    content = _load(self, data_dir, epoch, split)
    if split == "train":
        if len(content) < SHARD_LEN:
            raise ValueError(f"{split} shard for epoch {epoch} has {len(content)} < {SHARD_LEN} molecules")
        content = content[:SHARD_LEN]
        self._keys = self._keys[:SHARD_LEN]
    return content


_lmdb.LMDBDataset.load_data = _load_equal_length
