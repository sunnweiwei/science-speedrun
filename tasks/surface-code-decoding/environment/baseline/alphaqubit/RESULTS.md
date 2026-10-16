# AlphaQubit reproduction: results

Performance `P` is `1 − balanced_logical_failure`: round-count error fractions averaged within
each geometry/basis group, groups averaged within each distance, and d3 and d5 weighted equally.
The reward is `P`.

| run | P (reward) | d3 failure | d5 failure |
| --- | ---: | ---: | ---: |
| reproduced in this setup, from scratch with this recipe on H100s (10 groups × 18 h pretraining + 2 h fine-tuning, 200 GPU-h, 40 h wall on 8 GPUs) | **0.75569** | 0.24480 | 0.24382 |
| task author's frozen reference (about 200 GPU-h on RTX PRO 6000 Blackwell), replayed in this setup | 0.75507 | 0.24507 | 0.24479 |

Both were replayed by the task's verifier on all 3,250,000 test shots. The author's per-shot
validation predictions, validation metrics and training record are in
`/workspace/data/reference/results`; the reproduction's per-group training logs are in
`reproduction/run/`. No weights are shipped: every model you submit is trained from
random initialization in the session.
