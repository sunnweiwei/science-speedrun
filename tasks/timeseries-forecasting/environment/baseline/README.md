# Baselines

Reference implementations verified in this setup. Each one runs in this workspace and publishes a
package that follows `/workspace/contract.md`. `RESULTS.md` in each directory holds the numbers
we reproduced with the independent verifier. No weights are shipped.

| baseline | what it is | reward |
| --- | --- | ---: |
| [`seasonal_naive/`](seasonal_naive/) | repeat the last season; the reward's normalizer | 0.0075 |
| [`starter/`](starter/) | 6.4M-parameter patched transformer pretrained from scratch on the corpus (6 minutes on an H100) | 0.161 |
| [`starter-41m/`](starter-41m/) | the same recipe at 41.8M parameters, data-parallel on eight GPUs | 0.200 |
| [`timesfm3-50m/`](timesfm3-50m/) | **main baseline**: TimesFM-3's architecture (released code, random init) at 50.9M, source-balanced data mixture, eight GPUs, 3 h 14 min | **0.275** |

The best released zero-shot model (TimesFM-3, 331M, trained for days on 10^11 observations) reaches
0.447 under the same protocol (`/workspace/references/released_models.md`).
