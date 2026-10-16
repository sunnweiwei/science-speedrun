#!/bin/bash
#SBATCH --job-name=olmoearth-export-dispatch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
: "${OLMO_ROOT:?evaluation root required}" "${OLMO_NAME:?model name required}" "${OLMO_CHECKPOINT:?checkpoint required}" "${OLMO_SHARD_INPUTS:?input shards required}"
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
OLMO_PACKAGE="$OLMO_ROOT/packages/$OLMO_NAME"
if [[ -f "$OLMO_PACKAGE/provenance.json" ]]; then
  "$OLMO_UPSTREAM/.venv/bin/python" - "$OLMO_PACKAGE" "$OLMO_CHECKPOINT" <<'PY'
import json, sys
from pathlib import Path
provenance = json.loads((Path(sys.argv[1]) / "provenance.json").read_text())
if provenance.get("checkpoint") != sys.argv[2] or provenance.get("training_validity") == "invalid":
    raise ValueError("Existing package is not this valid checkpoint")
PY
else
  CUDA_VISIBLE_DEVICES="" "$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/environment/baseline/starter/export.py" \
    --upstream "$OLMO_UPSTREAM" --variant "${OLMO_VARIANT:-large}" --seed "${OLMO_SEED:-12536}" \
    --checkpoint "$OLMO_CHECKPOINT" --output "$OLMO_PACKAGE"
fi
"$OLMO_UPSTREAM/.venv/bin/python" "$OLMO_TRACK/slurm/submit_sharded_eval.py" \
  --root "$OLMO_ROOT" --name "$OLMO_NAME" --upstream "$OLMO_UPSTREAM" --shard-inputs "$OLMO_SHARD_INPUTS"
