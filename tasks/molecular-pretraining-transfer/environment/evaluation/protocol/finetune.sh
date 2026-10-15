#!/usr/bin/env bash
# Fixed fine-tune protocol for one encoder on one task: the official Uni-Mol2 fine-tune recipe (Uni-Mol 90f52c4
# unimol2/ README: lr 1e-4, batch 32, warmup 0.06, polynomial decay, smooth-MAE loss on normalised targets,
# test-time averaging over 11 conformers) for FT_EPOCHS epochs (default 15) with seed 0. The checkpoint with the
# best validation agg-MAE is kept and predicts the test split once (predict.py -> OUT/predictions.csv).
# usage: finetune.sh TASK DATA ENCODER OUT
#   DATA     directory with TASK/{train,valid,test}.lmdb and task_metainfo.json
#   ENCODER  a package directory (encoder.json + checkpoint.pt), or "none" for the untrained architecture
#   OUT      writable output directory
set -euo pipefail
task=$1; data=$2; enc=$3; out=$4
here=$(cd "$(dirname "$0")" && pwd)
num=$(python3 -c "import sys; sys.path.insert(0, '$here'); from tasks import NUM_TARGETS; print(NUM_TARGETS['$task'])")
arch=unimol2_84M; weights=()
if [ "$enc" != none ]; then
  arch=$(python3 -c "import json; print(json.load(open('$enc/encoder.json')).get('arch', 'unimol2_84M'))")
  ud=$(python3 -c "import json; print(json.load(open('$enc/encoder.json')).get('user_dir') or '')")
  [ -n "$ud" ] && export UNIMOL2X_SUBMISSION_USER_DIR=$enc/$ud
  weights=(--finetune-from-model "$enc/checkpoint.pt")
fi
export OMP_NUM_THREADS=1 PYTORCH_JIT=0 UNIMOL2X_METAINFO=$data/task_metainfo.json
mkdir -p "$out/tmp"; cd "$out"
common=(--task-name $task --user-dir $here/unimol2x --conf-size 11 --num-workers 4
        --task mol_finetune --loss finetune_smooth_mae --arch $arch
        --classification-head-name $task --num-classes $num --fp16 --reg --drop-feat-prob 1.0 --use-2d-pos-prob 0.0)
unicore-train "$data" "${common[@]}" --train-subset train --valid-subset valid --ddp-backend=c10d \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr 1e-4 --warmup-ratio 0.06 --max-epoch ${FT_EPOCHS:-15} \
    --batch-size 16 --update-freq 2 --pooler-dropout 0 --seed 0 \
    --fp16-init-scale 4 --fp16-scale-window 256 --log-interval 100 --log-format simple --validate-interval 1 \
    "${weights[@]}" \
    --best-checkpoint-metric valid_agg_mae --patience 20 --keep-last-epochs 1 --no-last-checkpoints \
    --no-save-optimizer-state --save-dir "$out/ckpt" --tmp-save-dir "$out/tmp" --find-unused-parameters
python3 "$here/predict.py" "$data" "${common[@]}" --valid-subset test --path "$out/ckpt/checkpoint_best.pt" \
    --batch-size 32 --seed 0 --predictions "$out/predictions.csv"
rm -rf "$out/ckpt" "$out/tmp"
