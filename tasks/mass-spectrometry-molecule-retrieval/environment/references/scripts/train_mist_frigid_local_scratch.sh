#!/usr/bin/env bash
# Stage tiny-file corpora into node-local storage, pack synthetic JSON, and train.
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 MIST_REPO MSG_ARCHIVE MSG_AUG_ARCHIVE OUTPUT_ROOT CUDA_DEVICE" >&2
  exit 2
fi

mist_repo="$1"
msg_archive="$2"
msg_aug_archive="$3"
output_root="$(mkdir -p "$4" && cd "$4" && pwd)"
cuda_device="$5"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
local_root="$(mktemp -d "${TMPDIR:-/tmp}/massspecgym-mist.${SLURM_JOB_ID:-local}.XXXXXX")"
trap 'rm -rf -- "$local_root"' EXIT

cp "$msg_archive" "$local_root/msg_preprocessed.tar.gz"
cp "$msg_aug_archive" "$local_root/msg_aug_filt.tar.gz"
printf '%s  %s\n' \
  44d3121313406e7f55a4c37ae714f4d3 "$local_root/msg_preprocessed.tar.gz" \
  32d8813d84d06364fc318e81f9cb4416 "$local_root/msg_aug_filt.tar.gz" \
  | md5sum --check --strict

mkdir -p "$local_root/frigid_msg"
decompressor=gzip
if command -v pigz >/dev/null 2>&1; then
  decompressor="pigz -p ${SLURM_CPUS_PER_TASK:-16}"
fi
tar -I "$decompressor" -xf "$local_root/msg_preprocessed.tar.gz" \
  -C "$local_root/frigid_msg"

python "$script_root/pack_synthetic_subformulae.py" \
  --archive "$local_root/msg_aug_filt.tar.gz" \
  --output-root "$local_root/msg_aug"
cp "$local_root/msg_aug/packing_report.json" "$output_root/packing_report.json"

mv "$local_root/msg_aug/labels.csv" "$local_root/msg_aug/labels.unfiltered.csv"
python "$script_root/filter_synthetic_heldout.py" \
  --labels "$local_root/frigid_msg/msg/labels.tsv" \
  --split "$local_root/frigid_msg/msg/split.tsv" \
  --synthetic-labels "$local_root/msg_aug/labels.unfiltered.csv" \
  --database "$local_root/msg_aug/subformulae.sqlite" \
  --output-labels "$local_root/msg_aug/labels.csv" \
  --output-report "$output_root/synthetic_filter_report.json"

python "$script_root/audit_training_data.py" \
  --labels "$local_root/frigid_msg/msg/labels.tsv" \
  --split "$local_root/frigid_msg/msg/split.tsv" \
  --synthetic-labels "$local_root/msg_aug/labels.csv" \
  --output "$output_root/training_data_audit.json" \
  --expected-synthetic-rows 2333485

bash "$script_root/train_mist_frigid.sh" \
  "$mist_repo" \
  "$local_root/frigid_msg/msg" \
  "$local_root/msg_aug" \
  "$output_root" \
  "$cuda_device"
