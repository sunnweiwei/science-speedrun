#!/bin/bash
#SBATCH --job-name=olmoearth-tiny
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=earth-%j.log
set -euo pipefail
[[ ${SLURM_JOB_PARTITION:-} == ${EARTH_SLURM_PARTITION:?Set EARTH_SLURM_PARTITION} ]] || exit 2
OLMO_UPSTREAM=${OLMO_UPSTREAM:-${EARTH_RUNTIME_ROOT:?Set EARTH_RUNTIME_ROOT}/olmoearth-upstream}
OLMO_DATA=${OLMO_DATA:-${EARTH_DATA_ROOT:?Set EARTH_DATA_ROOT}/olmoearth}
OLMO_TRACK=${OLMO_TRACK:-${EARTH_REPO_ROOT:?Set EARTH_REPO_ROOT}/earth-observation-pretraining-transfer}
OLMO_RUN=${OLMO_RUN:-$OLMO_DATA/runs/starter-${SLURM_JOB_ID}}
OLMO_STEPS=${OLMO_STEPS:-2000}
OLMO_LR=${OLMO_LR:-0.0000125}
OLMO_REPEATS=${OLMO_REPEATS:-32}
export OLMO_UPSTREAM
export OLMO_VARIANT=${OLMO_VARIANT:-tiny}
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
OLMO_CHECKPOINT_ARGS=()
if [[ -n ${OLMO_FIXED_CHECKPOINT_STEP:-} ]]; then
  [[ "$OLMO_FIXED_CHECKPOINT_STEP" =~ ^[0-9]+$ && ${OLMO_KEEP_CHECKPOINTS:-} =~ ^[0-9]+$ ]] || exit 2
  OLMO_CHECKPOINT_ARGS+=("--trainer.callbacks.checkpointer.fixed_steps=[$OLMO_FIXED_CHECKPOINT_STEP]")
  OLMO_CHECKPOINT_ARGS+=("--trainer.callbacks.checkpointer.max_checkpoints=$OLMO_KEEP_CHECKPOINTS")
fi
cd "$OLMO_UPSTREAM"
"$OLMO_UPSTREAM/.venv/bin/torchrun" --standalone --nproc_per_node="${OLMO_GPUS:-1}" "$OLMO_TRACK/environment/baseline/starter/train.py" train starter local \
  --common.save_folder="$OLMO_RUN" \
  --init_seed="${OLMO_SEED:-12536}" \
  --dataset.h5py_dir="${OLMO_PRETRAIN_DATA:-$OLMO_DATA/3980}" \
  --data_loader.global_batch_size="${OLMO_BATCH:-64}" --data_loader.num_workers=6 \
  --data_loader.num_dataset_repeats_per_epoch="$OLMO_REPEATS" \
  --data_loader.seed="${OLMO_DATA_SEED:-3622}" \
  --train_module.rank_microbatch_size="${OLMO_MICROBATCH:-32}" \
  --train_module.optim_config.lr="$OLMO_LR" \
  --train_module.optim_config.weight_decay="${OLMO_WEIGHT_DECAY:-0.02}" \
  --train_module.scheduler.warmup="${OLMO_WARMUP:-100}" \
  --trainer.load_strategy=never \
  --trainer.max_duration="{\"unit\":\"steps\",\"value\":$OLMO_STEPS}" \
  --trainer.callbacks.wandb.enabled=False \
  --trainer.callbacks.downstream_evaluator.enabled=False \
  --trainer.callbacks.checkpointer.save_interval="${OLMO_SAVE_INTERVAL:-500}" \
  --trainer.callbacks.checkpointer.ephemeral_save_interval="${OLMO_EPHEMERAL_INTERVAL:-250}" \
  "${OLMO_CHECKPOINT_ARGS[@]}"
