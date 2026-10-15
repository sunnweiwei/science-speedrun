#!/usr/bin/env bash
# Designs from a trained upstream-RFD3 checkpoint, in the task's submission format.
#
#   bash rfd3_infer.sh <ckpt> <view> <out> ["<tracks>"] [timesteps] [subset]
#   bash rfd3_infer.sh /workspace/research/ckpt/epoch-0060.ckpt /data/tune /tmp/run
#
# Three stages, with upstream's inference untouched in the middle:
#   rfd3_inputs.py   request .npz  -> RFD3 InputSpecification + a conditioning structure
#   rfd3 design      upstream, unmodified. Reduced widths come from the checkpoint's own
#                    `train_cfg`, so no config is repeated here.
#   rfd3_collect.py  output mmCIF  -> one PDB per design plus <track>.jsonl
#
# Then score it yourself - scoring is unrestricted, see evaluation/README.md:
#   python /workspace/evaluation/score.py --input <view> --designs <out>/submission
#
# That is the real designability metric; there is no geometry-only shortcut any more.
#
# Run that score from a normal working directory, NOT from /tmp: a phantom `matplotlib`
# directory there makes torchmetrics believe matplotlib is installed and then fail on
# `matplotlib.axes`. The full scorer imports rf3, so it inherits that trap.
set -euo pipefail

CKPT=${1:?checkpoint path}
VIEW=${2:?input view directory, e.g. /data/tune}
OUT=${3:?output directory}
TRACKS=${4:-"unconditional sm_binder ppi_binder enzyme"}
STEPS=${5:-200}
# Optional comma-separated example ids, for a cheap smoke run. Passed to hydra as
# a bracketed list: a bare `a,b` is rejected as an ambiguous value.
SUBSET=${6:-}

# Designs per target = N_BATCHES x 4 diffusion samples. The default 25 gives the 100
# designs per target that every task view requests and that RFdiffusion3's own protocol
# specifies - the same sample size behind the reference row in ../evaluation/RESULTS.md.
# Lower it only for a quick smoke test: at 4 designs per target the sm_binder and
# ppi_binder tracks return 0.000 whatever the model does, because their pass rates are a
# few percent over 4 and 5 targets. Budget ~110 s per design for the full scorer on top
# of generation.
N_BATCHES=${N_BATCHES:-25}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Dataloader workers bind an AF_UNIX socket under TMPDIR, and the path has a 108-char
# limit; a deep scratch path silently breaks inference rather than erroring clearly.
export TMPDIR=${TMPDIR:-/tmp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# `rfd3` is a console script installed by `pip install rc-foundry`. The image vendors
# the SOURCE and reaches it through PYTHONPATH instead, so that script does not exist
# there - invoke the same Typer app as a module when it is missing.
run_rfd3() {
  if command -v rfd3 >/dev/null 2>&1; then
    rfd3 "$@"
  else
    python -c 'from rfd3.cli import app; app()' "$@"
  fi
}

mkdir -p "$OUT"
python "$HERE/rfd3_inputs.py" --view "$VIEW" --out "$OUT/inputs" --tracks $TRACKS

for TRACK in $TRACKS; do
  [ -f "$OUT/inputs/$TRACK.json" ] || { echo "-- $TRACK: no targets, skipping"; continue; }
  echo "=== designing $TRACK ($((N_BATCHES * 4)) designs per target)"
  # A batch is 4 diffusion samples, so N_BATCHES=25 gives the 100 designs per target every
  # view requests while keeping peak memory at a single batch.
  run_rfd3 design \
    out_dir="$OUT/designs/$TRACK" \
    inputs="$OUT/inputs/$TRACK.json" \
    ckpt_path="$CKPT" \
    n_batches="$N_BATCHES" \
    diffusion_batch_size=4 \
    inference_sampler.num_timesteps="$STEPS" \
    skip_existing=True \
    prevalidate_inputs=True \
    ${SUBSET:+json_keys_subset=[$SUBSET]}
done

echo "=== collecting"
python "$HERE/rfd3_collect.py" --view "$VIEW" --designs "$OUT/designs" \
    --out "$OUT/submission" --tracks $TRACKS

echo "=== done: $OUT/submission"
