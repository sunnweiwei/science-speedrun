#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:small:1
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH -J rfd3-eval
#SBATCH -o %x_%j.log   # cd into $RFD3_EVAL_WORK/logs before submitting
#SBATCH --qos=normal
#
# Benchmark an upstream-RFD3 checkpoint on the protein-design task's four tracks.
#
#   sbatch run_eval.sh <run-name> <ckpt> <view-dir> "<tracks>" [num_timesteps] [subset]
#
# Stages: build RFD3 input specs from the view -> `rfd3 design` per track -> convert the
# designs into the task's submission format -> score designability in full. Scoring used
# to be `--geometry-only`, because designability needed AlphaFold 3 parameters absent from
# this host; RoseTTAFold3's are public, so the flag was removed on 2026-09-26 and this
# scores the published criterion. Set PROTEIN_DESIGN_ASSETS / LIGANDMPNN_ROOT if they are
# not at their image defaults.
set -euo pipefail

NAME=${1:?run name}
CKPT=${2:?checkpoint path}
VIEW=${3:?input view directory}
TRACKS=${4:-"unconditional sm_binder ppi_binder enzyme"}
STEPS=${5:-200}
# Optional comma-separated example ids, for smoke runs. Passed to hydra as a
# bracketed list: a bare `a,b` is rejected as an ambiguous value.
SUBSET=${6:-}

# Operator-side paths. Override by exporting them; the defaults are this checkout plus
# the host venv that carries rc-foundry, atomworks and lightning.
#
# sbatch copies the submitted script into the job spool directory, so BASH_SOURCE points
# at /var/slurm/spool/job<id>/ and does NOT locate the checkout. Slurm keeps the path it
# was submitted with, so recover it from there before falling back.
E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$E/make_inputs.py" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  E=$(dirname "$(scontrol show job "$SLURM_JOB_ID" \
        | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)")
fi
E=${PROTEIN_DESIGN_BENCH:-$E}
[ -f "$E/make_inputs.py" ] || { echo "cannot locate benchmark/ (got '$E'); set PROTEIN_DESIGN_BENCH" >&2; exit 2; }
# benchmark/ lives under environment/evaluation/, so score.py is its PARENT directory -
# do not rebuild the path from a task root.
EVAL=$(dirname "$E")
[ -f "$EVAL/score.py" ] || { echo "cannot locate evaluation/score.py next to '$E'" >&2; exit 2; }
VENV=${RFD3_VENV:-/net/tukwila/yanjing/rfd3-run/venv}
WORK=${RFD3_EVAL_WORK:-/net/tukwila/yanjing/rfd3-eval}
OUT=$WORK/runs/$NAME

# Dataloader workers bind an AF_UNIX socket in TMPDIR; a long path blows the 108-char
# limit, which is why this is not the session scratchpad.
export TMPDIR=/net/tukwila/yanjing/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$WORK/logs"

echo "=== $(date) $NAME: ckpt=$CKPT view=$VIEW tracks=[$TRACKS] steps=$STEPS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$VENV/bin/python $E/make_inputs.py --view "$VIEW" --out "$OUT/inputs" --tracks $TRACKS

for TRACK in $TRACKS; do
  [ -f "$OUT/inputs/$TRACK.json" ] || { echo "-- $TRACK: no targets, skipping"; continue; }
  echo "=== $(date) designing $TRACK ==="
  # num_designs is 100 per target in every view; 25 batches of 4 diffusion samples give
  # exactly that, and peak memory stays at one batch.
  $VENV/bin/rfd3 design \
    out_dir="$OUT/designs/$TRACK" \
    inputs="$OUT/inputs/$TRACK.json" \
    ckpt_path="$CKPT" \
    n_batches=25 \
    diffusion_batch_size=4 \
    inference_sampler.num_timesteps=$STEPS \
    skip_existing=True \
    prevalidate_inputs=True \
    ${SUBSET:+json_keys_subset=[$SUBSET]}
done

echo "=== $(date) collecting ==="
$VENV/bin/python $E/collect.py --view "$VIEW" --designs "$OUT/designs" \
    --out "$OUT/submission" --tracks $TRACKS

echo "=== $(date) scoring (designability) ==="
$VENV/bin/python $EVAL/score.py \
    --input "$VIEW" --designs "$OUT/submission" --tracks $TRACKS \
    | tee "$OUT/score.json"

echo "=== $(date) done: $OUT ==="
