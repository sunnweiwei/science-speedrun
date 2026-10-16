#!/usr/bin/env bash
# Usage: bash environment/data/materialize.sh release|rebuild TARGET_DIRECTORY
#   release  install the prepared HF release pinned in download-source.json (default path)
#   rebuild  rebuild everything from the upstream sources (see MATERIALIZE.md; about a day)
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
env_dir="$(cd "$script_dir/.." && pwd)"
[[ $# -eq 2 ]] || { sed -n '2,4p' "$0" >&2; exit 2; }
mode=$1; target=$2
image=${ATOMISTIC_MATERIALIZER_IMAGE:-atomistic-materials-env:v2}
[[ "${ATOMISTIC_MATERIALIZER_SKIP_BUILD:-0}" == 1 ]] || docker build -t "$image" "$env_dir"
mkdir -p "$(dirname "$target")"; parent="$(cd "$(dirname "$target")" && pwd)"; name="$(basename "$target")"
run() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$parent:/out" -v "$env_dir:/env:ro" \
          -w /env/data "$image" "$@"; }
case "$mode" in
  release)
    read -r repo rev sha < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["repo_id"], d["revision"], d["manifest_sha256"])' "$script_dir/download-source.json")
    run python release.py install --root "/out/$name" --repo-id "$repo" --revision "$rev" --manifest-sha256 "$sha"
    ;;
  rebuild)
    # v1 artifacts (splits, panels, test panel, 500k sample) come from the original task's pinned HF
    # bundle; everything else is rebuilt from upstream. Paths inside the container: /task/.data = v1.
    w="$parent/$name.build"; mkdir -p "$w/v1"
    rb() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$w:/w" -v "$w/v1:/task/.data" \
             -v "$env_dir:/task/environment:ro" -w /task/environment/data "$image" "$@"; }
    rb python materialize.py --root /task/.data --components training protocol initial-model eval-test trained-models
    rb bash fetch_pool.sh /w/pool-src
    rb python extra_suites.py --cache /w/cache --dev /w/dev --private /w/private
    rb bash -c 'cd ../evaluation && python panels.py --enlarge-v1 /task/.data/panels/v1/manifest.json --output /w/panels-v2.json &&
                python prepare.py --cache /w/sources --output /w/physical-v2-val --panel /w/panels-v2.json --preset val --workers 8'
    rb python build_pool.py --raw /w/pool-src/raw --omat /w/pool-src/omat-sel --extra /w/dev/extra /w/private/extra \
      --output /w/pool --workers "${ATOMISTIC_MATERIALIZER_WORKERS:-64}"
    rb python filter_standard.py --standard /task/.data/standard-v1/data --blocked /w/pool/blocked.json --output /w/mp-standard
    rb bash -c 'mkdir -p /w/baselines/mp-finetune-3ep && cp /task/.data/models/3ep/* /w/baselines/mp-finetune-3ep/ &&
                cp ../baseline/uma-mp-finetune/calculator.py /w/baselines/mp-finetune-3ep/'
    echo "Train baseline/uma-mixed-replay and export it to $w/baselines/mixed-replay before assembling." >&2
    rb python release.py assemble --pool /w/pool --physical-val /w/physical-v2-val \
      --physical-test /task/.data/physical-v1-test --extra-dev /w/dev/extra --extra-private /w/private/extra \
      --model /task/.data/models/uma-s-1p1.pt --mp-standard /w/mp-standard --baselines /w/baselines \
      --output "/w/release"
    mv "$w/release" "$parent/$name"
    ;;
  *) sed -n '2,4p' "$0" >&2; exit 2 ;;
esac
