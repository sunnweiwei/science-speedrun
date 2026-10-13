#!/usr/bin/env bash
# Materialize the public, held-out-clean MIST vFRIGID augmentation corpus.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 PUBLIC_DATA_ROOT MASSSPECGYM_1_5_TSV" >&2
  exit 2
fi

public_root="$(cd "$1" && pwd)"
source_tsv="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
reference_scripts="$script_root/../reference/scripts"
target="$public_root/development/synthetic_augmentation"
parent="$(dirname "$public_root")"
cache_root="${MASSSPECGYM_AUGMENTATION_CACHE_DIR:-$parent/massspecgym-augmentation-cache}"
archive="$cache_root/msg_aug_filt.tar.gz"

if [[ ! -f "$public_root/development/manifest.json" ]]; then
  echo "materialize the core MassSpecGym data before augmentation" >&2
  exit 1
fi
if [[ ! -f "$source_tsv" ]]; then
  echo "MassSpecGym source TSV not found: $source_tsv" >&2
  exit 1
fi
if [[ "$(sha256sum "$source_tsv" | cut -d' ' -f1)" != \
  50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a ]]; then
  echo "MassSpecGym source TSV checksum mismatch" >&2
  exit 1
fi
if [[ -e "$target" ]]; then
  python3 "$script_root/verify_synthetic_augmentation.py" "$target"
  printf 'verified existing clean synthetic augmentation at %s\n' "$target"
  exit 0
fi

mkdir -p "$cache_root"
if [[ ! -f "$archive" ]]; then
  curl -fL --retry 8 --retry-delay 3 --continue-at - \
    --output "$archive.part" \
    https://zenodo.org/api/records/20671842/files/msg_aug_filt.tar.gz/content
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' 32d8813d84d06364fc318e81f9cb4416 "$archive" \
  | md5sum --check --strict

stage_root="$(mktemp -d "$parent/.massspecgym-augmentation.XXXXXX")"
cleanup() {
  case "$stage_root" in
    "$parent"/.massspecgym-augmentation.*) rm -rf -- "$stage_root" ;;
    *) echo "refusing unsafe cleanup path: $stage_root" >&2 ;;
  esac
}
trap cleanup EXIT
mkdir -p "$stage_root/final"

python3 "$reference_scripts/pack_synthetic_subformulae.py" \
  --archive "$archive" \
  --output-root "$stage_root/packed"
mv "$stage_root/packed/labels.csv" "$stage_root/packed/labels.unfiltered.tsv"

python3 "$reference_scripts/filter_synthetic_heldout.py" \
  --experimental-tsv "$source_tsv" \
  --synthetic-labels "$stage_root/packed/labels.unfiltered.tsv" \
  --database "$stage_root/packed/subformulae.sqlite" \
  --output-labels "$stage_root/final/labels.tsv" \
  --output-report "$stage_root/final/sanitation_report.json"

python3 "$reference_scripts/audit_training_data.py" \
  --experimental-tsv "$source_tsv" \
  --synthetic-labels "$stage_root/final/labels.tsv" \
  --output "$stage_root/final/training_data_audit.json" \
  --expected-synthetic-rows 2333485

mv "$stage_root/packed/subformulae.sqlite" "$stage_root/final/subformulae.sqlite"
python3 - "$stage_root/final" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = {}
for name in ("labels.tsv", "subformulae.sqlite", "sanitation_report.json",
             "training_data_audit.json"):
    path = root / name
    files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
manifest = {
    "schema": "massspecgym-retrieval-synthetic-augmentation/v1",
    "source": {
        "zenodo_record": "20671842",
        "file": "msg_aug_filt.tar.gz",
        "md5": "32d8813d84d06364fc318e81f9cb4416",
        "license": "MIT",
    },
    "rows": 2333485,
    "files": files,
}
(root / "manifest.json").write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

chmod -R a+rX "$stage_root/final"
mv "$stage_root/final" "$target"
printf 'clean synthetic augmentation ready at %s\n' "$target"
