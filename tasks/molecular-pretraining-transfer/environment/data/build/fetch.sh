#!/usr/bin/env bash
# Download every upstream source at a pinned revision into SOURCES and check it against SOURCES.sha256.
# A file already present with the right sha256 is kept (so an existing download can be linked in).
# usage: fetch.sh SOURCES        needs curl, sha256sum, tar, python3 (+ pip for the ASAP export); network access
set -euo pipefail
S=$1; H=$(cd "$(dirname "$0")" && pwd); mkdir -p "$S"/{unimol2,compas,expansionrx,asap}; cd "$S"
UNIMOL2_REV=4748959698093f813731c04bfac8308031e2efdc      # HF dptech/unimol2-dataset (Apache-2.0), shards 0-11
COMPAS_REV=72e61e3d2bb1a22ecc333ad0e12b92b6ae6b6a04       # gitlab porannegroup/compas (CC BY 4.0)
EXRX_REV=6b898ccc43d10d25b230fb09e22a6e30c30022b5         # HF openadmet/openadmet-expansionrx-challenge-data (CC BY 4.0)
UNIMOL_FT=https://bioos-hermite-beijing.tos-cn-beijing.volces.com/unimol_data/finetune/molecular_property_prediction.tar.gz
ok() { [ -s "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$(grep "  $1\$" "$H/SOURCES.sha256" | cut -d' ' -f1)" ]; }
get() {  # get RELATIVE_PATH URL
  ok "$1" && { echo "ok   $1"; return; }
  echo "get  $1"; curl -sSfL --retry 5 -C - -o "$1.part" "$2" && mv "$1.part" "$1"
  ok "$1" || { echo "sha256 mismatch: $1" >&2; exit 1; }
}
rm -f .shard-failed
for p in $(seq 0 11); do   # four shards at a time; a failed or mismatched shard leaves .shard-failed
  ( get unimol2/train_part_$p.lmdb https://huggingface.co/datasets/dptech/unimol2-dataset/resolve/$UNIMOL2_REV/train/train_part_$p.lmdb \
    || touch .shard-failed ) &
  [ $((p % 4)) = 3 ] && wait
done; wait
[ ! -e .shard-failed ] || { echo "a pretraining shard failed to download or verify" >&2; exit 1; }
get molecular_property_prediction.tar.gz $UNIMOL_FT
[ -s molecular_property_prediction/raw_data/qm9.csv ] || tar -xzf molecular_property_prediction.tar.gz \
  molecular_property_prediction/qm9dft molecular_property_prediction/raw_data/qm9.csv
cp molecular_property_prediction/raw_data/qm9.csv qm9.csv
for f in compas-1D_b3lyp_def2-svp.csv compas-1D_b3lyp_def2-svp.tar.gz; do
  get compas/$f https://gitlab.com/porannegroup/compas/-/raw/$COMPAS_REV/COMPAS-1/$f; done
get compas/LICENSE https://gitlab.com/porannegroup/compas/-/raw/$COMPAS_REV/LICENSE
for f in expansion_data_train.csv expansion_data_test.csv expansion_data_raw.csv; do
  get expansionrx/$f https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data/resolve/$EXRX_REV/$f; done
# ASAP antiviral-ADMET-2025 (CC0) is distributed through Polaris; export it once with polaris-lib into a CSV.
f=asap/antiviral-admet-2025-unblinded.csv
if ! ok $f; then
  python3 -m venv /tmp/polaris-venv && /tmp/polaris-venv/bin/pip install -q "polaris-lib>=0.13" pandas pyarrow
  /tmp/polaris-venv/bin/python - <<'PY'
import pandas as pd, polaris as po
ds = po.load_dataset("asap-discovery/antiviral-admet-2025-unblinded")
pd.DataFrame([ds[i] for i in ds.rows]).to_csv("asap/antiviral-admet-2025-unblinded.csv", index=False)
PY
  ok $f || { echo "sha256 mismatch: $f" >&2; exit 1; }
fi
echo "all sources verified in $S"
