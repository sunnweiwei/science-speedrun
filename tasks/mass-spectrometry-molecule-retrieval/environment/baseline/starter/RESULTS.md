# Starter (DeepSets) and native GLACIER: reproduced results

Both parts of this directory were run from scratch in the task's development image and replayed
by the task's independent verifier on all 17,556 test queries.

| model | seeds | Hit@1 (reward) | Hit@5 | Hit@20 | MCES@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fourier-DeepSets starter (`README.md`, 50 epochs) | 1 | **4.004 %** | 12.36 % | 27.40 % | 22.56 |
| native GLACIER from scratch + DeepSets fallback (`glacier/README.md`) | 8 | **40.45 ± 0.36 %** | 56.91 % | 70.73 % | 13.19 |

- **Reward.** The reward is Hit@1. The official-weight GLACIER/MIST hybrid reaches 49.391 %
  (`/workspace/references/README.md`); it is a performance target, not a permitted initialization.
- **DeepSets.** The replay matched the task author's number exactly (4.004 %).
- **GLACIER.** The eight seeds range from 39.80 % to 41.11 % Hit@1; the task author's single run
  reached 40.32 %. Each seed took about 5 h of native training on one H100. Replay at inference
  batch 64 took 3.4–3.9 h on one H100 for all test queries.
- No weights are shipped.
