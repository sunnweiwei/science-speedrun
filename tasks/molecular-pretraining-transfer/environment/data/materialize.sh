#!/usr/bin/env bash
# Build the release from pinned upstream sources (MATERIALIZE.md). Run inside the task's runtime image, which has
# the pinned RDKit / LMDB / NumPy used to build it:
#   docker build -f environment/Dockerfile --target runtime -t mpt-runtime .        (context: the task directory)
#   docker run --rm -v "$PWD":/task:ro -v /large-disk/mpt:/out mpt-runtime bash /task/environment/data/materialize.sh /out
# usage: materialize.sh OUT [--verify-only]
#   OUT/sources/  downloads (fetch.sh; about 190 GB; may be deleted afterwards)
#   OUT/public/   pretrain/ (leakage-filtered corpus) and downstream/ (fine-tune data)   -> TASK_DATA_DIR=OUT
#   OUT/verifier/ test-inputs/ (test molecules without labels)                          -> TASK_PRIVATE_DIR=OUT
#   OUT/private/  labels/ (test labels, grader only)
set -euo pipefail
OUT=$(realpath "$1"); H=$(cd "$(dirname "$0")" && pwd); B=$H/build; P=$(nproc)
log() { echo "$(date -u +%FT%TZ) $*"; }
if [ "${2:-}" != --verify-only ]; then
  log "fetch"; bash "$B/fetch.sh" "$OUT/sources"
  if [ ! -s "$OUT/public/pretrain/valid_part_0.lmdb" ]; then
    log "leakage filter (12 shards)"
    python3 "$B/filter_shards.py" --src "$OUT/sources/unimol2" --downstream "$OUT/sources" --out "$OUT/public/pretrain.tmp" \
      --parts 0 1 2 3 4 5 6 7 8 9 10 11 --procs $((P > 150 ? 150 : P))
    mv "$OUT/public/pretrain.tmp" "$OUT/public/pretrain"
  fi
  rm -rf "$OUT/build"; mkdir -p "$OUT/build"
  log "downstream splits (QM9, COMPAS-1D)"
  python3 "$B/build_uni2_downstream.py" --unimol-ft "$OUT/sources/molecular_property_prediction" \
    --compas "$OUT/sources/compas" --out "$OUT/build/uni2" > "$OUT/build/uni2.log"
  log "evaluation suite"
  python3 "$B/build_fast.py" "$OUT/build/uni2" "$OUT/build/suite" full-test > "$OUT/build/suite.log"
  python3 "$B/build_admet.py" "$OUT/sources" "$OUT/build/suite" >> "$OUT/build/suite.log"
  log "views"
  rm -rf "$OUT/public/downstream" "$OUT/verifier" "$OUT/private"
  python3 "$B/make_views.py" "$OUT/build/suite" "$OUT/build/uni2/report.json" "$OUT"
  rm -rf "$OUT/build"   # holds test labels; only private/ keeps them
  chmod -R a+rX "$OUT/public" "$OUT/verifier" "$OUT/private"
fi
# Maintainers regenerate the pinned table with WRITE_CHECKSUMS=1 after a deliberate change to the build.
mode=verify; [ -n "${WRITE_CHECKSUMS:-}" ] && mode=write
log "$mode checksums"; python3 "$B/checksums.py" $mode "$OUT" "$H/CHECKSUMS.json" --procs $((P > 32 ? 32 : P))
log "done: TASK_DATA_DIR=TASK_PRIVATE_DIR=$OUT"
