#!/usr/bin/env bash
# Operator-side release builder. Usage:
#   materialize.sh SOURCE_CACHE DATA_ROOT [WORK_DIR]
# Downloads missing pinned sources into SOURCE_CACHE (network required for that step only),
# verifies them against download-source.json, then builds DATA_ROOT/{dev,private}.
# Stages needing arc-state / cell-eval / a GPU (embed, baseline) run in the task images:
#   VCZ_DEV_IMAGE (default vcz-dev:latest), VCZ_VERIFIER_IMAGE (default vcz-verifier:latest).
set -euo pipefail
cache=${1:?SOURCE_CACHE}; root=${2:?DATA_ROOT}; work=${3:-${root}.work}
here=$(cd "$(dirname "$0")" && pwd)
py=${VCZ_MATERIALIZE_PYTHON:-python3}
dev_image=${VCZ_DEV_IMAGE:-vcz-dev:latest}
verifier_image=${VCZ_VERIFIER_IMAGE:-vcz-verifier:latest}
docker_cmd=${VCZ_DOCKER:-docker}
mkdir -p "$cache" "$work" "$root"
cache=$(cd "$cache" && pwd); root=$(cd "$root" && pwd); work=$(cd "$work" && pwd)

if [[ "${VCZ_SKIP_DOWNLOAD:-0}" != "1" ]]; then
  "$py" "$here/download_sources.py" --manifest "$here/download-source.json" --cache "$cache"
fi
cd "$here"
"$py" -m materialize.cli verify --cache "$cache" --work "$work" ${VCZ_STRICT_VERIFY:+--strict}
"$py" -m materialize.cli panel --cache "$cache" --work "$work"
"$py" -m materialize.cli harmonize --cache "$cache" --work "$work" --workers "${VCZ_WORKERS:-32}"
"$py" -m materialize.cli build --cache "$cache" --work "$work" --root "$root"
"$py" -m materialize.cli priors --cache "$cache" --work "$work" --root "$root"
mkdir -p "$root/dev/se600m"
for f in se600m_epoch16.ckpt config.yaml protein_embeddings.pt MODEL_LICENSE.md MODEL_ACCEPTABLE_USE_POLICY.md; do
  [[ -e "$root/dev/se600m/$f" ]] || cp "$cache/se600m/$f" "$root/dev/se600m/$f"
done
# SE-600M embeddings of the shipped training cells (GPU, inside the dev image)
$docker_cmd run --rm --gpus all --shm-size 64g -v "$root:/release" -v "$here:/materialize:ro" -w /materialize \
  -e VCZ_GPUS="${VCZ_GPUS:-0,1,2,3}" --entrypoint python "$dev_image" \
  -m materialize.cli embed --cache /release/dev --work /release/dev --root /release --se-dir /release/dev/se600m --gpus "${VCZ_GPUS:-0,1,2,3}"
# no-effect baseline scores cached for the grader (cell-eval, inside the verifier image)
$docker_cmd run --rm -v "$root:/release" -v "$here:/materialize:ro" -v "$here/../evaluation:/evaluation:ro" \
  --entrypoint python "$verifier_image" /materialize/cache_noeffect_baseline.py --root /release
"$py" "$here/finalize_release.py" --root "$root"
echo "release ready: $root"
