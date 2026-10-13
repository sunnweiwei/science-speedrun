#!/usr/bin/env bash
# Download and verify the public GLACIER plus MIST fallback reference assets.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 ASSET_ROOT" >&2
  exit 2
fi

asset_root="$(mkdir -p "$1" && cd "$1" && pwd)"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$script_root/prepare_reference_assets.sh" "$asset_root"

ms_pred_commit=ed8311f22958cb37f055b663b5f56c5c77a2ee33
if [[ ! -d "$asset_root/ms-pred/.git" ]]; then
  git clone --filter=blob:none https://github.com/coleygroup/ms-pred.git \
    "$asset_root/ms-pred"
  git -C "$asset_root/ms-pred" checkout --detach "$ms_pred_commit"
elif [[ "$(git -C "$asset_root/ms-pred" rev-parse HEAD)" != "$ms_pred_commit" ]]; then
  echo "$asset_root/ms-pred exists at a different revision" >&2
  exit 1
fi
if ! git -C "$asset_root/ms-pred" diff --quiet || \
   ! git -C "$asset_root/ms-pred" diff --cached --quiet; then
  echo "$asset_root/ms-pred has tracked source modifications" >&2
  exit 1
fi

archive="$asset_root/downloads/GLACIER_checkpoint.zip"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 8 --retry-delay 3 --continue-at - \
    --output "$archive.part" \
    'https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui&dl=1'
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' \
  7e0fa24a19163513fe7cbeea3bf51a48cdd33bf2dbe8ac7a0457cbbf28d1258b \
  "$archive" | sha256sum --check --strict

checkpoint="$asset_root/checkpoints/glacier_msg_no_cf.ckpt"
if [[ ! -f "$checkpoint" ]]; then
  unzip -p "$archive" best.ckpt > "$checkpoint.part"
  mv "$checkpoint.part" "$checkpoint"
fi
printf '%s  %s\n' \
  5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11 \
  "$checkpoint" | sha256sum --check --strict

printf 'GLACIER hybrid reference assets verified at %s\n' "$asset_root"
