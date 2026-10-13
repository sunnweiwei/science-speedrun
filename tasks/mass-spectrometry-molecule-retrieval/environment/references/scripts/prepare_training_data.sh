#!/usr/bin/env bash
# Download and verify the exact processed corpora used to train MIST vFRIGID.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 DATA_ROOT" >&2
  exit 2
fi

data_root="$(mkdir -p "$1" && cd "$1" && pwd)"
download_root="$data_root/downloads"
mkdir -p "$download_root" "$data_root/frigid_msg" "$data_root/frigid_aug"

fetch() {
  local url="$1" destination="$2" expected_md5="$3"
  if [[ ! -f "$destination" ]]; then
    curl -fL --retry 8 --retry-delay 3 --continue-at - \
      --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
  fi
  local actual_md5
  actual_md5="$(md5sum "$destination" | cut -d' ' -f1)"
  if [[ "$actual_md5" != "$expected_md5" ]]; then
    echo "md5 mismatch for $destination: $actual_md5" >&2
    exit 1
  fi
}

extract_archive() {
  local archive="$1" destination="$2"
  local decompressor="gzip"
  if command -v pigz >/dev/null 2>&1; then
    decompressor="pigz -p ${PIGZ_THREADS:-16}"
  fi
  tar -I "$decompressor" -xf "$archive" -C "$destination"
}

extract_synthetic_archive() {
  local archive="$1" destination="$2"
  local decompressor="gzip"
  if command -v pigz >/dev/null 2>&1; then
    decompressor="pigz -p ${PIGZ_THREADS:-16}"
  fi
  # The archive contains 2.33 million JSON files in one directory. Some shared
  # filesystems cap a single directory before that point. FRIGID discovers
  # augmentation files recursively, so route the archive's existing 000--010
  # filename partitions into subdirectories without changing any contents.
  local transforms=() part
  for part_number in {0..10}; do
    printf -v part '%03d' "$part_number"
    transforms+=(
      --transform="s#^msg_synth_filt/subformulae/msg_synth_filt_part_${part}_#msg_synth_filt/subformulae/part_${part}/msg_synth_filt_part_${part}_#"
    )
  done
  tar -I "$decompressor" "${transforms[@]}" -xf "$archive" -C "$destination"
}

msg_archive="$download_root/msg_preprocessed.tar.gz"
aug_archive="$download_root/msg_aug_filt.tar.gz"

fetch \
  https://zenodo.org/api/records/15008938/files/msg_preprocessed.tar.gz/content \
  "$msg_archive" 44d3121313406e7f55a4c37ae714f4d3
fetch \
  https://zenodo.org/api/records/20671842/files/msg_aug_filt.tar.gz/content \
  "$aug_archive" 32d8813d84d06364fc318e81f9cb4416

if [[ ! -f "$data_root/frigid_msg/msg/labels.tsv" ]]; then
  extract_archive "$msg_archive" "$data_root/frigid_msg"
fi
if [[ ! -f "$data_root/frigid_aug/msg_synth_filt/labels.csv" ]]; then
  extract_synthetic_archive "$aug_archive" "$data_root/frigid_aug"
fi

if [[ "$(wc -l < "$data_root/frigid_msg/msg/labels.tsv")" -ne 231105 ]]; then
  echo "unexpected experimental labels row count" >&2
  exit 1
fi
if [[ "$(wc -l < "$data_root/frigid_aug/msg_synth_filt/labels.csv")" -ne 2333511 ]]; then
  echo "unexpected synthetic labels row count" >&2
  exit 1
fi
synthetic_json_count="$({ find "$data_root/frigid_aug/msg_synth_filt/subformulae" \
  -type f -name '*.json' -printf '.\n'; } | wc -l)"
if [[ "$synthetic_json_count" -ne 2333510 ]]; then
  echo "unexpected synthetic subformula JSON count: $synthetic_json_count" >&2
  exit 1
fi

printf 'training data verified at %s\n' "$data_root"
printf 'MSG_ROOT=%s\n' "$data_root/frigid_msg/msg"
printf 'MSG_AUG_ROOT=%s\n' "$data_root/frigid_aug/msg_synth_filt"
