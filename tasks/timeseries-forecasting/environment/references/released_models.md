# Released models under this task's protocol

The same 97 GIFT-Eval configurations, metrics and normalizer as the reward. The "task replay" column
is the task author's replay of each released checkpoint through this task's verifier. The other
columns are the public leaderboard's own numbers. The integrator has not re-run these. Replay
and board agree to within 0.003 reward except for Chronos-Bolt (0.006, from a changed long-horizon
heuristic in its library). None of these checkpoints is in the workspace, and none may be used.

| system | relative MASE | relative WQL | reward (board) | reward (task replay) | how it was trained |
| --- | ---: | ---: | ---: | ---: | --- |
| Seasonal Naive | 1.000 | 1.000 | 0.000 | 0.0075 | none |
| PatchTST | 0.849 | 0.587 | 0.294 | | one model per dataset, supervised on GIFT-Eval train splits |
| chronos_bolt_base (205M) | 0.808 | 0.574 | 0.319 | 0.3254 | pretrained; the board flags test leakage |
| TimesFM-2.5 (231M) | 0.705 | 0.490 | 0.412 | 0.4121 | zero-shot |
| Chronos-2 (119M) | 0.698 | 0.485 | 0.418 | 0.4180 | pretrained |
| **TimesFM-3 (331M)** | 0.667 | 0.456 | 0.449 | **0.4465** (rel MASE 0.6685, rel WQL 0.4583) | zero-shot; the best released zero-shot model on the board |

The released models were pretrained for days on multi-GPU clusters, on corpora of about 10^11
observations. The task author also trained a 41.8M-parameter scale-up of the starter recipe on
this corpus: reward 0.207 (not reproduced by the integrator).
