#!/usr/bin/env bash
#SBATCH --job-name=qec-replay
#SBATCH --account=users
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|v100
#SBATCH --exclude=gpu-compute5,linux[41-60]
#SBATCH --array=0-9%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source "${QEC_TASK_ROOT}/slurm/runtime.sh"
: "${QEC_CAMPAIGN:=native-v1}"
qec_group="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${QEC_TASK_ROOT}/reproduction/groups.txt")"
for qec_stage in pretrain finetune; do
  qec_package="${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/replay/${qec_stage}"
  cd "${qec_package}"
  # Override the original workspace path and CUDA-library locations explicitly.
  export PYTHONPATH="${qec_package}"
  qec_replay_cuda="$("${qec_package}/.venv/bin/python" -c 'import pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()["purelib"]); print(":".join(str(x) for x in sorted((p/"nvidia").glob("*/lib"))))')"
  export LD_LIBRARY_PATH="${qec_replay_cuda}"
  "${qec_package}/.venv/bin/python" -m baselines.predict \
    --model "${qec_package}/models/${qec_group}/model.keras" \
    --data-root "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/replay-inputs" \
    --group "${qec_group}" --split test --require-gpu \
    --output "${QEC_ASSET_ROOT}/${QEC_CAMPAIGN}/predictions/${qec_stage}/${qec_group}"
done
