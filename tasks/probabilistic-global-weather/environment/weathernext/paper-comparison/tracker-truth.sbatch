#!/bin/bash
#SBATCH --job-name=wn2-paper-tracker-truth
#SBATCH --account=users
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=1-7
set -euo pipefail
: "${ROOT:?}" "${SOURCE:?}" "${PYTHON:?}" "${TRACKER_BIN:?}"
index=${SLURM_ARRAY_TASK_ID:?}
printf -v case_stem 'case-%03d' "$index"
initialization=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_initializations"][int(sys.argv[2])])' "$SOURCE/paper-comparison/weather-protocol.json" "$index")
exec "$PYTHON" "$SOURCE/paper-comparison/run_tracker.py" \
    --bin "$TRACKER_BIN" --input "$ROOT/tracker/truth/$case_stem.nc" \
    --initialization "$initialization" --output "$ROOT/tracker/observed-$case_stem"
