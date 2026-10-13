# B0 no effect: reproduced results

Release v1, cell-eval 0.8.2, isolated verifier (three test contexts) and `run_loco.py` (four
training-context folds). Record: `../evaluation_history.jsonl`, candidate `B0-noeffect`.

| reward (final) | leaderboard | near final | mid final | far final | LOCO k562 | jurkat | hct116 | hek293t |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |

Every per-metric normalised score is 0 by construction (differences are below 1e-16).
