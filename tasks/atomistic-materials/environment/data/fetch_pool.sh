#!/usr/bin/env bash
# Download the pool's upstream sources into POOL_ROOT/raw and the OMat24 shard selection into
# POOL_ROOT/omat-sel. OMat24 subsets are fetched one at a time; a deterministic 35 % of each subset's
# ASE-LMDB shards (SHA-256 of "omat-pool-v1:<member>" mod 100 < 35) is kept and the tar deleted.
# About 225 GB of downloads; peak disk about 120 GB. Rerunning skips finished subsets.
set -euo pipefail
P=${1:?usage: fetch_pool.sh POOL_ROOT}
B=https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train
mkdir -p "$P/raw" "$P/omat-sel"; cd "$P/raw"
[ -s mptrj.json ] || curl -fsSL -o mptrj.json https://ndownloader.figshare.com/files/41619375
[ -s salex-train.tar.gz ] || curl -fsSL -o salex-train.tar.gz \
  https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz
[ -s matpes-pbe.parquet ] || curl -fsSL -o matpes-pbe.parquet \
  "https://huggingface.co/datasets/colabfit/MatPES-PBE-2025.1/resolve/main/co/co_0.parquet"
for s in rattled-300-subsampled rattled-500-subsampled rattled-1000-subsampled rattled-300 rattled-500 \
         rattled-1000 rattled-relax aimd-from-PBE-1000-npt aimd-from-PBE-1000-nvt aimd-from-PBE-3000-npt \
         aimd-from-PBE-3000-nvt; do
  [ -f "$P/omat-sel/$s.done" ] && continue
  [ -f "omat-$s.tar.gz" ] || curl -fsSL -o "omat-$s.tar.gz" "$B/$s.tar.gz"
  tar -tzf "omat-$s.tar.gz" | grep 'aselmdb$' | python3 -c "
import sys, hashlib
for m in (l.strip() for l in sys.stdin):
    if int(hashlib.sha256(('omat-pool-v1:' + m).encode()).hexdigest(), 16) % 100 < 35: print(m)" \
    > "$P/omat-sel/$s.members"
  tar -xzf "omat-$s.tar.gz" -C "$P/omat-sel" -T "$P/omat-sel/$s.members"
  rm -f "omat-$s.tar.gz"; touch "$P/omat-sel/$s.done"
done
