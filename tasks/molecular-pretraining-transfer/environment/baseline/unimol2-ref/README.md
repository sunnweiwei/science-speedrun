# unimol2-ref: official Uni-Mol2 84M, paper recipe, shortened schedule

The official Uni-Mol2 code (Uni-Mol 90f52c4 `unimol2/`, Uni-Core ace6fae, installed in this image), pretrained from
scratch on `/workspace/data/pretrain` with the recipe of the paper (arXiv 2406.14969, Sec. 4 and appendix):

- `unimol2_84M`: 12 layers, 768 atom / 512 pair dimensions, 48 heads; 83.4 M encoder parameters;
- masked-token, coordinate and distance losses 1:1:1; 15 % masking; coordinate noise normal(0, 0.2); feature drop 0.5;
- Adam (0.9, 0.99, eps 1e-6), weight decay 1e-4, clip 1.0, lr 1e-4 with polynomial decay; batch 1,024 molecules;
  fp16.

Only the schedule is shortened: **30,000 updates with 3,000 warmup** instead of the paper's long run. That is about
30.7 M molecules seen (about three of the eleven shards used), 5.1 h on 8 H100 at about 1.65 updates/s.

```bash
bash /workspace/baseline/unimol2-ref/pretrain.sh                  # -> /workspace/research/unimol2-ref/{ckpt,train.log}
python /workspace/baseline/unimol2-ref/publish.py /workspace/research/unimol2-ref/ckpt/checkpoint_last.pt
python /workspace/evaluation/dev_eval.py --encoder /workspace/submission --out /workspace/research/eval/unimol2-ref
```

`pretrain.sh` sets the gradient accumulation so that the batch is 1,024 for any GPU count (`NGPU`, `LOCAL_BS`). It
resumes from `checkpoint_last.pt` after a failure, up to five attempts. `MAX_UPDATE`, `WARMUP` and `SAVE_EVERY`
change the schedule.

Two things differ from running the official scripts directly:

- **Equal shard lengths** (`unimol2x/`). The official `LMDBDataset` switches shards every epoch but keeps the batch
  sampler of the first epoch. The leakage-filtered shards differ by a few molecules, so without this fix one rank asks
  for an index past the end of a shorter shard in epoch 2. The `IndexError` is then hidden by Uni-Core's final
  barrier and shows up as an NCCL hang. The wrapper truncates every training shard to 10,485,658 molecules, and
  `pretrain.sh` rotates epochs over shards 0–10. Shard 11 is left out because it lost the validation hold-out.
- **NCCL timeout.** Uni-Core hard-codes a 90 s collective timeout. The image raises it to 30 min, so one slow rank
  on a shared host does not kill the run.

`publish.py` keeps only the model weights: `checkpoint.pt` is 0.3 GB instead of 1 GB with optimizer state. It
writes `encoder.json`, runs `/workspace/evaluation/check_submission.py` and swaps the package in atomically. Results:
[RESULTS.md](RESULTS.md).
