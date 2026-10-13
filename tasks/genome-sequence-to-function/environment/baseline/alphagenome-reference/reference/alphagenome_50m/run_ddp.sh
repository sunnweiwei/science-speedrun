#!/usr/bin/env bash
# run_ddp.sh DATA_DIR OUTPUT_DIR CONFIG EPOCHS [GPUS=8] [WORKERS_PER_GPU=12]
# Train the reference on every train interval with one example per GPU per step, then score
# validation with the verifier's own scorer. Runs inside the task image (torch, torchrun).
set -Eeuo pipefail
DATA=$1; OUT=$2; CONFIG=$3; EPOCHS=$4; GPUS=${5:-8}; WORKERS=${6:-12}
BASELINE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
cd "$BASELINE"
# Loopback rendezvous: works offline, where --standalone cannot resolve the hostname.
torchrun --nnodes 1 --nproc-per-node "$GPUS" --master-addr 127.0.0.1 --master-port "${MASTER_PORT:-29511}" -m reference.alphagenome_50m.train_eval train \
  --input "$DATA" --config "$CONFIG" --output "$OUT" --epochs "$EPOCHS" --workers "$WORKERS" \
  --log-every 200 --save-every 2000 2>&1 | tee -a "$OUT/train.log"
python predict.py "$DATA" "$OUT/valid-pred" --checkpoint "$OUT/last.pt" --split valid
python /workspace/evaluation/score.py --predictions "$OUT/valid-pred/predictions.zarr" \
  --labels "$DATA" --split valid --output "$OUT/valid-score"
