# Starter: reproduced results

The four commands in `README.md` (full-tier pack, 30,000 steps, seed 0, KernelSynth mix 0.1) were
run in the task's development image on one H100:

- packing the corpus took 1.5 minutes;
- training took 6 minutes;
- the best validation checkpoint was published and replayed by the task's independent verifier
  on all 97 GIFT-Eval configurations (326,490 instances, 7 minutes).

| | relative MASE | relative WQL | reward |
| --- | ---: | ---: | ---: |
| **starter (this directory)** | 1.006 | 0.699 | **0.161** |
| short-term configs (55) | 0.877 | 0.652 | |
| medium-term configs (21) | 1.181 | 0.774 | |
| long-term configs (21) | 1.231 | 0.760 | |

Reward is `clip(1 − sqrt(rel MASE × rel WQL), 0, 1)` over the 97 configurations. It is ahead of
Seasonal Naive on short horizons and behind it on MASE at medium and long horizons. On the
verifier's normalized scale (Seasonal Naive 0, the best released zero-shot model 1) it reads
0.314. Validation skill during training was −0.054 on 15 stratified configurations. That is a
different scale from the reward (`README.md`).

The task author's run of the same recipe (A100-40GB, 16 minutes) scored 0.153. The difference is
one seed on different hardware; no seed band is claimed. The weights are not shipped.

For scale:

| system | reward |
| --- | ---: |
| Seasonal Naive (`../seasonal_naive/`) | 0.0075 |
| starter | 0.161 |
| TimesFM-3, best released zero-shot model (`/workspace/references/released_models.md`) | 0.447 |
