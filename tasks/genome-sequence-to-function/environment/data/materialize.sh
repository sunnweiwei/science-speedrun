#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
task_root="$(cd "$script_dir/../.." && pwd)"
image="${ALPHAGENOME_MATERIALIZER_IMAGE:-genome-sequence-to-function-materializer:latest}"

usage() {
  cat >&2 <<'EOF'
Usage: bash environment/data/materialize.sh MODE TARGET_DIRECTORY

Builds the release from AlphaGenome's public TFRecords (FOLD_0): every eligible train interval
(41,020, 8-bit targets), 200 validation and 1,000 test intervals (full precision), the pinned
GRCh38.p13 reference and the variant evaluations. See MATERIALIZE.md for costs.

Modes:
  release       Install the prepared release from Hugging Face (repo, revision and manifest SHA-256
                pinned in download-source.json); every archive is verified, then the full validator runs
  rebuild       Rebuild from upstream instead: bootstrap, reference, index, full, variant-evals
  bootstrap     Download pinned metadata and write manifests
  reference     Download and verify the pinned GRCh38.p13 FASTA and index
  inspect       Inventory every source bundle
  probe         Validate the first aligned VALID record in all bundles
  smoke         Transform one real aligned VALID record
  index         Freeze source-backed train/valid/test selections
  full          Materialize and export all indexed splits, then validate
  variant-evals Fetch the pinned evaluation tables and build the perturbation queries and labels
  validate      Validate a complete materialized release

ALPHAGENOME_MATERIALIZER_WORKERS (default 64) sets the number of streaming workers per split.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage
mode="$1"
destination="$2"
case "$mode" in bootstrap|reference|inspect|probe|smoke|index|full|variant-evals|release|rebuild|validate) ;; *) usage ;; esac
command -v docker >/dev/null || { echo 'ERROR: docker is required.' >&2; exit 2; }
mkdir -p -- "$destination"
destination="$(cd "$destination" && pwd)"

if [[ "${ALPHAGENOME_MATERIALIZER_SKIP_BUILD:-0}" != "1" ]]; then
  docker build --pull -f "$script_dir/materializer.Dockerfile" -t "$image" "$task_root"
fi
docker_limits=()
if [[ -n "${ALPHAGENOME_MATERIALIZER_MEMORY:-}" ]]; then
  docker_limits+=(--memory "$ALPHAGENOME_MATERIALIZER_MEMORY")
fi
if [[ -n "${ALPHAGENOME_MATERIALIZER_MEMORY_SWAP:-}" ]]; then
  docker_limits+=(--memory-swap "$ALPHAGENOME_MATERIALIZER_MEMORY_SWAP")
fi
if [[ -n "${ALPHAGENOME_MATERIALIZER_CPUS:-}" ]]; then
  docker_limits+=(--cpus "$ALPHAGENOME_MATERIALIZER_CPUS")
fi
run_agbench() {
  docker run --rm "${docker_limits[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$destination:/output" "$image" "$@"
}

workers="${ALPHAGENOME_MATERIALIZER_WORKERS:-64}"

materialize_all() {
  run_agbench fetch-reference --data-dir /output
  # Each split is streamed shard by shard (resumable) and exported (manifests, labels, scoring).
  for split in train valid test; do
    run_agbench materialize --data-dir /output --split "$split" --workers "$workers"
  done
  run_agbench validate --data-dir /output --splits train,valid,test
}

variant_evals() {
  local sources="$destination/variant_eval_sources"
  bash "$script_dir/fetch_variant_evals.sh" "$sources"
  docker run --rm "${docker_limits[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$destination:/output" -v "$script_dir:/build:ro" --entrypoint python "$image" \
    /build/build_variant_evals.py --sources /output/variant_eval_sources --release /output \
    --output /output/variant_evals
}

case "$mode" in
  bootstrap) run_agbench bootstrap --data-dir /output ;;
  reference) run_agbench fetch-reference --data-dir /output ;;
  inspect)
    run_agbench bootstrap --data-dir /output
    run_agbench inspect --data-dir /output --splits train,valid,test
    ;;
  probe)
    run_agbench bootstrap --data-dir /output
    run_agbench inspect --data-dir /output --splits valid --probe
    ;;
  smoke)
    run_agbench bootstrap --data-dir /output
    run_agbench inspect --data-dir /output --splits valid --probe
    run_agbench smoke --data-dir /output --split valid
    run_agbench smoke --data-dir /output --split train
    ;;
  index)
    run_agbench bootstrap --data-dir /output
    run_agbench index --data-dir /output --splits train,valid,test
    ;;
  full)
    for split in valid train; do
      test -s "$destination/smoke/SMOKE_${split^^}_VALIDATED.json" || {
        echo "ERROR: run $split smoke before full materialization." >&2; exit 2;
      }
    done
    test -s "$destination/metadata/source_index_report.json" || {
      echo 'ERROR: run index before full materialization.' >&2; exit 2;
    }
    materialize_all
    ;;
  variant-evals) variant_evals ;;
  release)
    mapfile -t pinned < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); [print(d[k] or "") for k in ("repo_id","revision","manifest_sha256")]' "$script_dir/download-source.json")
    [[ -n "${pinned[1]}" && -n "${pinned[2]}" ]] || { echo "ERROR: download-source.json has no pinned revision; use rebuild." >&2; exit 2; }
    # The installer writes a sibling temporary directory, then renames it into place.
    parent="$(dirname "$destination")"; name="$(basename "$destination")"
    rmdir "$destination" 2>/dev/null || true
    docker run --rm "${docker_limits[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$parent:/install" "$image" install-hf --data-dir "/install/$name" \
      --repo-id "${pinned[0]}" --revision "${pinned[1]}" --manifest-sha256 "${pinned[2]}"
    ;;
  rebuild)
    run_agbench bootstrap --data-dir /output
    run_agbench inspect --data-dir /output --splits valid --probe
    run_agbench smoke --data-dir /output --split valid
    run_agbench smoke --data-dir /output --split train
    run_agbench index --data-dir /output --splits train,valid,test
    materialize_all
    variant_evals
    ;;
  validate) run_agbench validate --data-dir /output --splits train,valid,test ;;
esac
