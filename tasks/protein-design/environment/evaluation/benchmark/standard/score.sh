#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:small:1
#SBATCH --mem=48g
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -J std-score
#SBATCH -o score_%A_%a.log   # cd into $STANDARD_BENCH_WORK/logs before submitting
#SBATCH --qos=normal
#
# Stage 3 of the standard benchmark: inverse-fold + refold + judge, one target per task.
#
#   TRACK=enzyme STANDARD_BENCH_WORK=$W \
#     sbatch --array=0-40 --dependency=afterany:<genjob> score.sh
#
# -t 24:00:00 IS DELIBERATE - do not lower it. Scoring runs 7-11 h per target (~200
# s/design x 100 on the slower ones) and score_standard.py writes its single JSON only
# at the end, so a timeout loses the ENTIRE target; there is no per-design checkpoint.
# At 12 h, 3 of 37 enzyme targets timed out. `scontrol update jobid=... TimeLimit=` is
# admin-only on this cluster, so a running job cannot be rescued - only resubmitted.
#
# Use --dependency=afterANY, not afterok. Two reasons: afterok on an array needs ALL
# tasks to succeed, so one failed generation leaves this job unstartable and eventually
# killed as DependencyNeverSatisfied; and for a rerun after a timeout afterok can never
# be satisfied at all. The skip guard below makes afterany safe - a target that already
# has a score file costs one second, not a GPU-hour - so the same script serves as both
# the first pass and the rerun.
set -euo pipefail

TRACK=${TRACK:?track (unconditional|sm_binder|ppi_binder|enzyme)}

E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ ! -f "$E/score_standard.py" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  E=$(dirname "$(scontrol show job "$SLURM_JOB_ID" \
        | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)")
fi
E=${PROTEIN_DESIGN_BENCH_STANDARD:-$E}
[ -f "$E/score_standard.py" ] || {
  echo "cannot locate benchmark/standard (got '$E'); set PROTEIN_DESIGN_BENCH_STANDARD" >&2
  exit 2
}

VENV=${RFD3_VENV:-/net/tukwila/yanjing/rfd3-run/venv}
W=${STANDARD_BENCH_WORK:?work dir}
ASSETS=${DESIGNABILITY_ASSETS:-/net/tukwila/yanjing/designability-assets}
LIGANDMPNN=${LIGANDMPNN_DIR:-/net/tukwila/yanjing/ligandmpnn}
SPECS=$W/inputs/$TRACK.json
[ -f "$SPECS" ] || { echo "no specs at $SPECS" >&2; exit 2; }

export TMPDIR=${RFD3_TMPDIR:-/net/tukwila/yanjing/tmp}
mkdir -p "$TMPDIR" "$W/scores" "$W/logs"

# Run from the work dir, NOT /tmp: a phantom `matplotlib` directory there makes
# torchmetrics believe matplotlib is installed and then fail on `matplotlib.axes`.
cd "$W"

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
  echo "no SLURM_ARRAY_TASK_ID - scoring all $N targets sequentially"
  SELECTED=("${TARGETS[@]}")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for TARGET in "${SELECTED[@]}"; do
  OUT=$W/scores/$TRACK-$TARGET.json
  # The rerun guard. A non-empty score file means this target already finished, so the
  # same array can be resubmitted after a timeout without re-spending its GPU-hours.
  if [ -s "$OUT" ]; then
    echo "score file already present for $TARGET - nothing to do"
    continue
  fi
  echo "=== $(date) scoring $TRACK / $TARGET ==="
  "$VENV/bin/python" "$E/score_standard.py" \
    --designs "$W/designs/$TRACK" --track "$TRACK" --target "$TARGET" \
    --out "$OUT" \
    --assets "$ASSETS" \
    --ligandmpnn "$LIGANDMPNN"
  echo "=== $(date) done $TARGET ==="
done
