#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:small:1
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH -J std-gen
#SBATCH -o gen_%A_%a.log   # cd into $STANDARD_BENCH_WORK/logs before submitting
#SBATCH --qos=normal
#
# Stage 2 of the standard benchmark: generate 100 designs per target.
#
#   TRACK=enzyme CKPT=<ckpt> STANDARD_BENCH_WORK=$W sbatch --array=0-40 generate.sh
#
# ONE ARRAY TASK PER TARGET, not per track. The enzyme track is 41 targets x 100
# designs; in a single job that exceeds any sane walltime. The target is selected out of
# the stage-1 spec file by $SLURM_ARRAY_TASK_ID and passed to hydra as
# `json_keys_subset=[<target>]` - a bare `a,b` is rejected as an ambiguous value.
#
# Set --array to 0-(N-1) where N is the track's target count; the script prints N and
# refuses an index past the end, so a wrong range fails loudly instead of silently
# skipping targets. With no SLURM_ARRAY_TASK_ID it runs every target sequentially, which
# is what you want for the 1-target unconditional track or a run without slurm.
#
# Stage 1 is make_specs.py, stage 3 score.sh, stage 4 report.py. See README.md.
set -euo pipefail

TRACK=${TRACK:?track (unconditional|sm_binder|ppi_binder|enzyme)}
CKPT=${CKPT:?checkpoint path}
STEPS=${STEPS:-200}

# sbatch copies the submitted script into the job spool directory, so BASH_SOURCE points
# at /var/slurm/spool/job<id>/ and does NOT locate the checkout. Slurm keeps the path it
# was submitted with, so recover it from there before falling back.
E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$E/make_specs.py" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  E=$(dirname "$(scontrol show job "$SLURM_JOB_ID" \
        | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)")
fi
E=${PROTEIN_DESIGN_BENCH_STANDARD:-$E}
[ -f "$E/make_specs.py" ] || {
  echo "cannot locate benchmark/standard (got '$E'); set PROTEIN_DESIGN_BENCH_STANDARD" >&2
  exit 2
}

VENV=${RFD3_VENV:-/net/tukwila/yanjing/rfd3-run/venv}
W=${STANDARD_BENCH_WORK:?work dir (must be FRESH - see below)}
SPECS=$W/inputs/$TRACK.json
[ -f "$SPECS" ] || { echo "no specs at $SPECS - run make_specs.py first" >&2; exit 2; }

# Dataloader workers bind an AF_UNIX socket in TMPDIR; a long path blows the 108-char
# limit, which is why this is not a session scratchpad.
export TMPDIR=${RFD3_TMPDIR:-/net/tukwila/yanjing/tmp}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$W/designs/$TRACK" "$W/logs"

# The image installs no `rfd3` console script - the source is vendored and reached by
# PYTHONPATH - so fall back to the module entry point.
if [ -x "$VENV/bin/rfd3" ]; then RFD3=("$VENV/bin/rfd3")
else RFD3=("$VENV/bin/python" -c 'from rfd3.cli import app; app()'); fi

mapfile -t TARGETS < <("$VENV/bin/python" -c "
import json,sys
print('\n'.join(sorted(json.load(open(sys.argv[1])))))" "$SPECS")
N=${#TARGETS[@]}
echo "track $TRACK: $N targets in $SPECS"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID past the end ($N targets); use --array=0-$((N-1))" >&2
    exit 2
  fi
  SELECTED=("${TARGETS[$SLURM_ARRAY_TASK_ID]}")
else
  echo "no SLURM_ARRAY_TASK_ID - running all $N targets sequentially"
  SELECTED=("${TARGETS[@]}")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for TARGET in "${SELECTED[@]}"; do
  echo "=== $(date) generating $TRACK/$TARGET from $CKPT ==="
  # 25 batches x 4 diffusion samples = 100 designs.
  #
  # skip_existing=True makes a populated designs/ a silent no-op, which is why the work
  # dir must be FRESH per checkpoint: reusing one scores the PREVIOUS checkpoint's
  # designs under the new label, and nothing warns.
  "${RFD3[@]}" design \
    inputs="$SPECS" \
    out_dir="$W/designs/$TRACK" \
    ckpt_path="$CKPT" \
    n_batches=25 \
    diffusion_batch_size=4 \
    inference_sampler.num_timesteps="$STEPS" \
    skip_existing=True \
    prevalidate_inputs=True \
    json_keys_subset="[$TARGET]"
  echo "=== $(date) done $TARGET: $(ls "$W/designs/$TRACK/$TARGET"* 2>/dev/null | wc -l) files ==="
done
