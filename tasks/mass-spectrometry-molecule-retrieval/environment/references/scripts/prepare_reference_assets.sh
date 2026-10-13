#!/usr/bin/env bash
# Download and verify the public assets needed for reference inference.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 ASSET_ROOT" >&2
  exit 2
fi

asset_root="$(mkdir -p "$1" && cd "$1" && pwd)"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$asset_root/checkpoints" "$asset_root/downloads"

fetch() {
  local url="$1" destination="$2" algorithm="$3" expected="$4"
  if [[ ! -f "$destination" ]]; then
    curl -fL --retry 8 --retry-delay 3 --continue-at - \
      --output "$destination.part" "$url"
    mv "$destination.part" "$destination"
  fi
  local actual
  actual="$("${algorithm}sum" "$destination" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$algorithm mismatch for $destination: $actual" >&2
    exit 1
  fi
}

clone_at() {
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
    git -C "$destination" checkout --detach "$commit"
  elif [[ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]]; then
    echo "$destination exists at a different revision; refusing to modify it" >&2
    exit 1
  fi
}

clone_at \
  https://github.com/samgoldman97/mist-cf.git \
  "$asset_root/mist-cf" \
  71c820b43825e0b0a58680a3690f29a7352290f4
clone_at \
  https://github.com/coleygroup/FRIGID.git \
  "$asset_root/FRIGID" \
  5e537525a9ae26208efcf882a07e3b8df2413c48

if git -C "$asset_root/mist-cf" apply --check \
  "$script_root/../patches/mist-cf-lightning2.patch" 2>/dev/null; then
  git -C "$asset_root/mist-cf" apply \
    "$script_root/../patches/mist-cf-lightning2.patch"
elif ! git -C "$asset_root/mist-cf" apply --reverse --check \
  "$script_root/../patches/mist-cf-lightning2.patch" 2>/dev/null; then
  echo "MIST-CF compatibility patch neither applies nor is already present" >&2
  exit 1
fi

if git -C "$asset_root/FRIGID" apply --check \
  "$script_root/../patches/mist-frigid-deterministic-seed.patch" 2>/dev/null; then
  git -C "$asset_root/FRIGID" apply \
    "$script_root/../patches/mist-frigid-deterministic-seed.patch"
elif ! git -C "$asset_root/FRIGID" apply --reverse --check \
  "$script_root/../patches/mist-frigid-deterministic-seed.patch" 2>/dev/null; then
  echo "MIST deterministic-seed patch neither applies nor is already present" >&2
  exit 1
fi

if git -C "$asset_root/FRIGID" apply --check \
  "$script_root/../patches/mist-frigid-sqlite-augmentation.patch" 2>/dev/null; then
  git -C "$asset_root/FRIGID" apply \
    "$script_root/../patches/mist-frigid-sqlite-augmentation.patch"
elif ! git -C "$asset_root/FRIGID" apply --reverse --check \
  "$script_root/../patches/mist-frigid-sqlite-augmentation.patch" 2>/dev/null; then
  echo "MIST SQLite-augmentation patch neither applies nor is already present" >&2
  exit 1
fi

(
  cd "$asset_root/mist-cf"
  printf '%s  %s\n' \
    84568c6ae5660dcfbfb93be4574a8cd73ef16cfd3d12577473af0adefb9517a1 src/mist_cf/common/__init__.py \
    419002f37a3c3ad52770f62cf78013bd038cac004722c00c08df8000c59c30bc src/mist_cf/common/misc_utils.py \
    | sha256sum --check --strict
)
(
  cd "$asset_root/FRIGID"
  printf '%s  %s\n' \
    e919e1a2ebd7ad32abf9ed88d2dd95c4983db3b5e4bba4dd1e2c94840adf8433 src/mist/data/featurizers.py \
    4db21f3c2f2276b06280ba153ffde7d128cabaf2fc6dc095b00f60264853e664 src/mist/utils/misc_utils.py \
    | sha256sum --check --strict
)

hf_root="https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/resolve/d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a/models"
fetch "$hf_root/mist_cf_msg.ckpt" \
  "$asset_root/checkpoints/mist_cf_msg.ckpt" sha256 \
  53345744860a92ed2c3cb4a9326313d1d50d38011e178f33e02a12ac0dbea729
fetch "$hf_root/mist_cf_fast_filter.ckpt" \
  "$asset_root/checkpoints/mist_cf_fast_filter.ckpt" sha256 \
  7e26802251551a23904cd57f3f2f691dfd4aefc8fd4c8fba8e5ee2059c624791

fetch \
  https://zenodo.org/api/records/19685145/files/frigid_pretrained_checkpoints.tar.gz/content \
  "$asset_root/downloads/frigid_pretrained_checkpoints.tar.gz" md5 \
  1059c193b5f3dd7034079076e943389b
if [[ ! -f "$asset_root/checkpoints/mist_msg.pt" ]]; then
  tar -xzf "$asset_root/downloads/frigid_pretrained_checkpoints.tar.gz" \
    -C "$asset_root/checkpoints" mist_msg.pt
fi
if [[ "$(sha256sum "$asset_root/checkpoints/mist_msg.pt" | cut -d' ' -f1)" != \
  09b4e93e8d5d35e7472f4d187d72ca316b125368232bb69148df101ec211b92a ]]; then
  echo "sha256 mismatch for mist_msg.pt" >&2
  exit 1
fi

fetch \
  https://github.com/sirius-ms/sirius/releases/download/v5.5.7/sirius-5.5.7-linux64.zip \
  "$asset_root/downloads/sirius-5.5.7-linux64.zip" sha256 \
  e3d21669cbd896649ee3ddea565e69070fa124de32a8386bdd6b203c9c0b304c
if [[ ! -x "$asset_root/sirius/bin/sirius" ]]; then
  unzip -q "$asset_root/downloads/sirius-5.5.7-linux64.zip" -d "$asset_root"
fi

printf 'reference assets verified at %s\n' "$asset_root"
