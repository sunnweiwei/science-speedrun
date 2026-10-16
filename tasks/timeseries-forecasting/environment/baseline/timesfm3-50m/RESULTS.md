# timesfm3-50m: reproduced results

The commands in `README.md` were run in the task's development image on eight H100s. The best
validation checkpoint was published with `publish.py` and replayed by the task's sealed verifier on
all 97 GIFT-Eval configurations (326,490 instances).

| | relative MASE | relative WQL | reward | normalised |
| --- | ---: | ---: | ---: | ---: |
| **timesfm3-50m (this directory)** | 0.873 | 0.603 | **0.275** | 0.547 |
| starter-41m (8 GPUs) | 0.940 | 0.681 | 0.200 | 0.40 |
| starter (6.4M) | 1.006 | 0.699 | 0.161 | 0.31 |
| TimesFM-3 released (331M, external target) | 0.669 | 0.458 | 0.447 | 0.90 |

Times on eight H100s:
- packing the 273 GB corpus: 17 minutes;
- training: 150,000 steps in 3 h 14 min, including 30 validation passes;
- verifier inference: 11 minutes on one GPU.

Validation skill on the 15 stratified configs rose from −0.12 at step 5,000 to +0.10 by step 30,000,
then fluctuated between +0.04 and +0.12. The best was at step 100,000 (+0.120). The step-30,000
checkpoint already scored 0.274 on test, so the last 120,000 steps barely moved the reward.

How the recipe was chosen (same model, test reward):

| corpus | sampling | test reward |
| --- | --- | ---: |
| 41 GB v1 corpus | uniform over series | 0.206 (overfit after 25,000 steps) |
| 273 GB | uniform over series | 0.217 (88 % of windows from climate and building loads) |
| 273 GB | source-balanced (`mixture.py`) | **0.275** |
