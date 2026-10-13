#!/usr/bin/env bash
# B2 training: STATE ST (state_sm) in gene space — STATE's own VCC recipe: cell sets of log1p
# CP10k expression on the common panel in, predicted expression out — with ESM-2 perturbation
# features. (The SE-600M-latent variant of ST needs Arc's SE counts decoder, whose checkpoint
# paths are hard-coded in the pinned arc-state; see README.) Usage: train.sh [max_steps] [overrides...]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
steps=${1:-20000}; shift || true
export WANDB_MODE=disabled
state tx train \
  data.kwargs.toml_config_path="$here/st_data/zeroshot.toml" \
  data.kwargs.output_space=all \
  data.kwargs.pert_col=gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.batch_col=gem_group \
  data.kwargs.control_pert=non-targeting \
  data.kwargs.perturbation_features_file="$here/st_data/esm2_pert_features.pt" \
  data.kwargs.num_workers=8 \
  model=state_sm \
  training.max_steps="$steps" \
  training.batch_size=16 \
  training.val_freq=1000000 \
  training.ckpt_every_n_steps=2000 \
  use_wandb=false training.wandb_track=false \
  output_dir="$here/runs" \
  name=zeroshot "$@"
