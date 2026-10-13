#!/usr/bin/env bash
# Reference solution: the reference AlphaQubit recipe from scratch (no trained weights are supplied).
# Per group: pretrain on pij simulations, then fine-tune on the training rows, selected on validation;
# groups run in parallel, one per visible GPU. The recorded reference took ~200 GPU-hours
# (RTX PRO 6000 Blackwell); QEC_PRETRAIN_HOURS / QEC_FINETUNE_HOURS scale it to the session budget.
set -euo pipefail
cd /workspace/baseline/alphaqubit
export PYTHONPATH=/workspace/baseline/alphaqubit XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
MODELS=${QEC_MODELS:-/workspace/research/models}; PRE=${QEC_PRETRAIN_HOURS:-18}; FT=${QEC_FINETUNE_HOURS:-2}
mapfile -t GROUP_LIST < reproduction/groups.txt  # not GROUPS: a read-only bash variable (mapfile fails under set -e)
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
common=(--data-root /workspace/data --seed 0 --dtype bfloat16 --microbatch 256 --batch-change-counter steps --eval-shots 5120 --eval-seconds 600)
train_group() {  # group gpu
  local g=$1; export CUDA_VISIBLE_DEVICES=$2
  python -m baselines.alphaqubit_train "${common[@]}" --group "$g" --stage pretrain --hours "$PRE" --output "$MODELS/$g/pretrain"
  python -m baselines.alphaqubit_train "${common[@]}" --group "$g" --stage finetune --hours "$FT" \
    --init-model "$MODELS/$g/pretrain/best.npz" --output "$MODELS/$g/finetune"
}
i=0
for g in "${GROUP_LIST[@]}"; do
  train_group "$g" "${GPUS[$((i % ${#GPUS[@]}))]}" > "$MODELS.$g.log" 2>&1 &
  i=$((i + 1))
  if (( i % ${#GPUS[@]} == 0 )); then wait; fi
done
wait
python -m baselines.alphaqubit_artifact --models-root "$MODELS" --stage finetune --output /workspace/research/frozen
python -m qecbench.package --package /workspace/research/frozen --output /workspace/submission
