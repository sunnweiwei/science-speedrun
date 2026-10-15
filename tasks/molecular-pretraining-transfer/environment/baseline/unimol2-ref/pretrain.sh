#!/usr/bin/env bash
# Reference pretraining: the official Uni-Mol2 84M (Uni-Mol 90f52c4 unimol2/, Uni-Core ace6fae) from scratch on
# /workspace/data/pretrain with the paper's recipe (arXiv 2406.14969): losses token:coord:dist = 1:1:1, mask 15 %,
# coordinate noise normal(0, 0.2), drop_feat 0.5, Adam(0.9, 0.99, eps 1e-6), weight decay 1e-4, clip 1.0, lr 1e-4,
# polynomial decay, batch 1024 molecules, fp16. Only the schedule is shortened: MAX_UPDATE (default 30,000) with
# WARMUP (default 3,000) updates, 5.1 h on 8 H100.
# usage: pretrain.sh [RUN_DIR]   (default /workspace/research/unimol2-ref); env NGPU, LOCAL_BS, MAX_UPDATE, WARMUP,
#        SAVE_EVERY. The run resumes from RUN_DIR/ckpt/checkpoint_last.pt (up to five attempts).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=${1:-/workspace/research/unimol2-ref}; data=$run/data; mkdir -p "$data" "$run/ckpt/tmp"
NG=${NGPU:-$(nvidia-smi -L | grep -c '^GPU ')}; BS=${LOCAL_BS:-32}; UF=$((1024 / (NG * BS)))
[ $((NG * BS * UF)) = 1024 ] || { echo "NGPU x LOCAL_BS must divide 1024" >&2; exit 1; }
# The official loader reads epoch e from /temp/train_part_{e-1}.lmdb (copying it there from the data directory when it is
# missing). Epoch e is linked to shard (e-1) % 11 (shard 11 holds the validation hold-out and is left out, see
# unimol2x), and /temp points to the link directory so nothing is copied.
for e in $(seq 0 59); do ln -sfn /workspace/data/pretrain/train_part_$((e % 11)).lmdb "$data/train_part_$e.lmdb"; done
ln -sfn /workspace/data/pretrain/valid_part_0.lmdb "$data/valid_part_0.lmdb"
rm -rf /temp && ln -sfn "$data" /temp
export OMP_NUM_THREADS=1 PYTORCH_JIT=0
cd /opt/Uni-Mol/unimol2
for attempt in 1 2 3 4 5; do
  echo "=== attempt $attempt $(date -u +%FT%TZ)" >> "$run/train.log"
  torchrun --nnodes 1 --nproc_per_node $NG --rdzv-backend c10d --rdzv-endpoint 127.0.0.1:${PORT:-29511} \
    $(which unicore-train) "$data" --user-dir "$here/unimol2x" --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=c10d --task unimol2 --loss unimol2 --arch unimol2_84M --mode train \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay 1e-4 \
    --lr-scheduler polynomial_decay --lr 1e-4 --warmup-updates ${WARMUP:-3000} --total-num-update ${MAX_UPDATE:-30000} \
    --update-freq $UF --seed 1 --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --max-update ${MAX_UPDATE:-30000} --max-epoch 60 --log-interval 100 --log-format simple \
    --save-interval-updates ${SAVE_EVERY:-5000} --keep-interval-updates 3 --no-epoch-checkpoints \
    --validate-interval-updates ${SAVE_EVERY:-5000} \
    --masked-token-loss 1 --masked-coord-loss 1 --masked-dist-loss 1 \
    --mask-prob 0.15 --drop-feat-prob 0.5 --noise-type normal --noise 0.2 \
    --batch-size $BS --save-dir "$run/ckpt" --tmp-save-dir "$run/ckpt/tmp" >> "$run/train.log" 2>&1 && exit 0
  sleep 60
done
exit 1
