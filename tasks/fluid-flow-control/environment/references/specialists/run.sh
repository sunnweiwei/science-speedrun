#!/usr/bin/env bash
# Manual baseline invocation, not an agent harness. All remaining args go to native.py.
# Usage: bash run.sh RUN_NAME GPU_INDEX train --algo PPO --seed 0 --steps 40000
set -euo pipefail
run_name="${1:?run name required}"
gpu_index="${2:?GPU index required}"
shift 2
[[ "$run_name" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 2
[[ "$gpu_index" =~ ^[0-7]$ ]] || exit 2
baseline_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
resource_root="${HYDROGYM_REPRO_ROOT:-/mnt/workspace/hydrogym}"
data_root="${HYDROGYM_BASELINE_DATA_DIR:-$resource_root/data}"
test -d "$data_root/environments"
test -d "$resource_root/paper/source/hydrogym-1.0.1"
test "$(git -C "$resource_root/paper/source/TheHydroGymProject" rev-parse HEAD)" = 20a483ecc6ea711f6b380fff05f212ca4d7e2c67
test -z "$(git -C "$resource_root/paper/source/TheHydroGymProject" status --porcelain --untracked-files=no)"
test ! -e "$resource_root/reproduction/$run_name"
mkdir -p "$resource_root/reproduction/$run_name/source"
cp -r "$baseline_dir/recipes" "$resource_root/reproduction/$run_name/source/recipes"
evaluator_dir="$baseline_dir/evaluator"
if [[ ! -d "$evaluator_dir" ]]; then evaluator_dir="$baseline_dir/../../evaluation"; fi
cp -r "$evaluator_dir" "$resource_root/reproduction/$run_name/source/evaluator"
cp "$baseline_dir/native.py" "$baseline_dir/learner.py" "$baseline_dir/run.sh" \
  "$baseline_dir/public_learner.py" "$baseline_dir/Dockerfile" \
  "$baseline_dir/pinball_author.py" \
  "$baseline_dir/time_limits.py" \
  "$baseline_dir/PUBLIC_REPRODUCTION.md" "$baseline_dir/test_public_learner.py" \
  "$resource_root/reproduction/$run_name/source/"
docker run -d --name "$run_name" --user 0 --network none --gpus "device=$gpu_index" \
  --shm-size=16g \
  --mount "type=bind,source=$data_root,target=/data,readonly" \
  --mount "type=bind,source=$resource_root/paper/source/hydrogym-1.0.1,target=/upstream,readonly" \
  --mount "type=bind,source=$resource_root/paper/source/TheHydroGymProject,target=/author-source,readonly" \
  --mount "type=bind,source=$resource_root/reproduction/$run_name/source,target=/baselines,readonly" \
  --mount "type=bind,source=$resource_root/reproduction,target=/runs" \
  --env PYTHONPATH=/upstream --env OMPI_MCA_coll_hcoll_enable=0 \
  --entrypoint bash ai4sci-hydrogym-baselines:20260905 -lc \
  'set -e; source /etc/profile; module purge; module load MAIA/1.0-NVHPC-26.1; source /home/easybuild/venvs/hydrogym_gpu/bin/activate; exec python /baselines/native.py "$@"' \
  hydrogym-baseline "$@" --output "/runs/$run_name"
