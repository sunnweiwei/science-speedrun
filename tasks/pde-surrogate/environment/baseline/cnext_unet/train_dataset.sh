#!/usr/bin/env bash
# Train The Well's CNextU-net recipe from scratch on one dataset.
#   train_dataset.sh <dataset_dir_name> <data yaml name> [epochs]
# Datasets: turbulent_radiative_layer_2D and active_matter (The Well layout under /workspace/data/well-<name>)
# and pdearena_navier_stokes_2d (written by convert_pdearena_to_well.py under $CNEXT_WORK/well-<name>).
# The Well's source (its Hydra configs are not in the wheel) is in the image at /opt/the_well, commit
# ad50de0e8861380d7c4d62c0503c654710f026e7. +model.gradient_checkpointing=true keeps the published batch size
# (64) within 40 GB; it changes memory and speed, not the optimization.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")   # before the cd below, so a relative $0 still resolves
DS=$1; CFG=$2; EPOCHS=${3:-200}
DATA=${DATA_DIR:-/workspace/data}
SRC=${THE_WELL_SRC:-/opt/the_well}
WORK=${CNEXT_WORK:-/workspace/research/cnext_unet}
RUNS=${WELL_RUNS:-$WORK/runs}; LOG=${WELL_LOG:-$WORK/logs}
mkdir -p "$LOG" "$RUNS"
# The Well's loader wants <base>/<dataset>/data/{train,valid,test}. The data mounts are read-only and hold no
# test split (the held-out cases stay with the verifier), so a writable base of symlinks is assembled here with
# test pointing at valid: the trainer's closing test pass then reports validation numbers, and the task's
# own score comes only from the independent replay. Hydra writes outputs/ into the cwd, so the trainer runs
# from the runs directory.
BASE="$WORK/well_base"; mkdir -p "$BASE"
for d in "$DATA"/well-*/ "$WORK"/well-*/; do
  [ -d "$d" ] || continue
  n=$(basename "$d"); n=${n#well-}; mkdir -p "$BASE/$n/data"
  for f in "$d"*; do [ "$(basename "$f")" = data ] || ln -sfn "$f" "$BASE/$n/$(basename "$f")"; done
  for s in train valid test; do
    if [ -d "$d/data/$s" ]; then ln -sfn "$d/data/$s" "$BASE/$n/data/$s"
    elif [ "$s" = test ]; then ln -sfn "$d/data/valid" "$BASE/$n/data/test"; fi
  done
done
[ -e "$BASE/$DS/data/train" ] || { echo "no dataset $DS: expected $DATA/well-$DS or $WORK/well-$DS (convert PDEArena first)" >&2; exit 1; }
cd "$RUNS"
export WANDB_MODE=offline WANDB_SILENT=true HYDRA_FULL_ERROR=1 PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
# datasets converted into The Well's layout are not in its name list, so they are registered first (see register_local_dataset.py)
RUNNER=(python3 -m the_well.benchmark.train)
if [ -n "${WELL_EXTRA_DATASETS:-}" ]; then
  RUNNER=(python3 "$HERE/register_local_dataset.py" -m the_well.benchmark.train)
fi
timeout "${WELL_TIME_CAP:-43200}" "${RUNNER[@]}" experiment=unet_convnext data="$CFG" server=local \
  data.well_dataset_name="$DS" data.well_base_path="$BASE" experiment_dir="$RUNS" \
  trainer.epochs="$EPOCHS" data_workers="${WELL_WORKERS:-6}" +model.gradient_checkpointing=true ${WELL_EXTRA:-} > "$LOG/well_cnext_$DS.log" 2>&1
rc=$?
if [ $rc -ne 0 ] || grep -aq "Traceback" "$LOG/well_cnext_$DS.log" || ! find "$RUNS" -name '*.pt' -newer "$LOG/well_cnext_$DS.log" -o -name '*.pt' | grep -q .; then
  echo "[$(date +%FT%T)] $DS FAILED rc=$rc" >> "$LOG/well_cnext_queue.log"; exit 1
fi
echo "[$(date +%FT%T)] $DS finished rc=$rc" >> "$LOG/well_cnext_queue.log"
