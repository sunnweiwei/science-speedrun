#!/usr/bin/env bash
# Reproduce the released FRIGID MassSpecGym MIST fingerprint encoder.
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 MIST_REPO MSG_ROOT MSG_AUG_ROOT OUTPUT_ROOT CUDA_DEVICE" >&2
  exit 2
fi

mist_repo="$1"
msg_root="$2"
msg_aug_root="$3"
output_root="$4"
cuda_device="$5"

export CUDA_VISIBLE_DEVICES="$cuda_device"

if [[ -f "$msg_aug_root/subformulae.sqlite" ]]; then
  forward_aug_source="$msg_aug_root/subformulae.sqlite"
else
  forward_aug_source="$msg_aug_root/subformulae"
fi

cd "$mist_repo"

python "$mist_repo/src/mist/train_mist.py" \
  --cache-featurizers \
  --name msg_mist \
  --labels-file "$msg_root/labels.tsv" \
  --subform-folder "$msg_root/subformulae/default_subformulae" \
  --spec-folder "$msg_root/spec_files" \
  --fp-names morgan4096 \
  --num-workers 16 \
  --seed 1 \
  --gpus 1 \
  --batch-size 256 \
  --iterative-preds growing \
  --iterative-loss-weight 0.4 \
  --learning-rate 0.00077 \
  --weight-decay 1e-07 \
  --lr-decay-frac 0.9 \
  --hidden-size 640 \
  --pairwise-featurization \
  --peak-attn-layers 2 \
  --refine-layers 4 \
  --spectra-dropout 0.1 \
  --split-file "$msg_root/split.tsv" \
  --form-embedder pos-cos \
  --no-diffs \
  --save-dir "$output_root" \
  --shuffle-train \
  --augment-data \
  --frac-orig 0.08 \
  --cosine-schedule \
  --max-epochs 150 \
  --warmup-frac 0.1 \
  --forward-labels "$msg_aug_root/labels.csv" \
  --forward-aug-folder "$forward_aug_source" \
  --ema \
  --ema-decay 0.995 \
  --max-peaks 10
