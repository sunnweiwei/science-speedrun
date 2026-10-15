#!/usr/bin/env bash
# Prepare both data views. Fails closed when a pinned source is missing.
#
#   bash materialize.sh SOURCE_DIR [TARGETS_DIR] OUTPUT_ROOT [extra materialize.py args]
#
# SOURCE_DIR   archived bulk indexes: entries.idx, clusters-by-entity-30.txt
# TARGETS_DIR  optional; defaults to the benchmark targets vendored in this repository
# OUTPUT_ROOT  destination for evaluation/ and development/
set -euo pipefail

usage() {
    echo 'Usage: bash materialize.sh SOURCE_DIR [TARGETS_DIR] OUTPUT_ROOT [args...]' >&2
    echo '  SOURCE_DIR must contain entries.idx and clusters-by-entity-30.txt' >&2
    echo '  See environment/data/README.md for the download commands.' >&2
    exit 2
}

[[ $# -lt 2 ]] && usage
here="$(cd "$(dirname "$0")" && pwd)"
sources="$1"
if [[ $# -ge 3 && -d "$2" ]]; then
    targets="$2"; output="$3"; shift 3
else
    targets="${here}/targets"; output="$2"; shift 2
fi

index="${sources}/entries.idx"
clusters="${sources}/clusters-by-entity-30.txt"
for required in "$index" "$clusters"; do
    if [[ ! -s "$required" ]]; then
        echo "Missing pinned source: $required" >&2
        echo 'See environment/data/README.md for the download commands.' >&2
        exit 2
    fi
done
for track in enzyme sm ppi; do
    if [[ ! -s "${targets}/${track}/input.yaml" ]]; then
        echo "Missing benchmark targets: ${targets}/${track}/input.yaml" >&2
        exit 2
    fi
done
# Both are needed before any work starts: gemmi for every structure read, yaml for the
# ppi target spec. Checking yaml only when ingest_targets.py reaches it would fail the
# run several minutes in, after the source checks have already passed.
python3 -c 'import gemmi' 2>/dev/null || {
    echo 'gemmi is required on the host: pip install gemmi' >&2; exit 2; }
python3 -c 'import yaml' 2>/dev/null || {
    echo 'pyyaml is required on the host: pip install pyyaml' >&2; exit 2; }

echo '==> Evaluation view (curated benchmark targets)'
python3 "${here}/ingest_targets.py" --results-root "$targets" \
    --output-root "${output}/evaluation"

echo '==> Development views (dated PDB snapshot)'
python3 "${here}/materialize.py" --pdb-index "$index" --clusters "$clusters" \
    --assembly-cache "${sources}/rcsb-cache" --output-root "$output" \
    --skip-evaluation "$@"

echo "Prepared under ${output}"
