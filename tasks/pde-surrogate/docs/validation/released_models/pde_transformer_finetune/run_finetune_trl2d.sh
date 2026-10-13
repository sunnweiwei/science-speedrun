#!/usr/bin/env bash
# Launch upstream's trainer on the converted turbulent-layer split. Usage: run_finetune_trl2d.sh [extra dotlist overrides]
set -uo pipefail
UP=${PDET_UPSTREAM_DIR:?set PDET_UPSTREAM_DIR to a clone of tum-pbs/pde-transformer, the paper trainer}
REF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p $UP/config/finetune && cp $REF/finetune_trl2d.yaml $UP/config/finetune/ && cp $REF/env_local.yaml $UP/env/pde_surrogate.yaml
cd $UP && WANDB_MODE=offline WANDB_SILENT=true python3 main.py -c config/finetune/finetune_trl2d.yaml --env env/pde_surrogate.yaml --logdir /home/deqing_google_com/data/pde-surrogate/finetune/logs --dryrun "$@"
