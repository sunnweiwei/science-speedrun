# Reproduction record: B1 / B1g mean delta

The leave-one-context-out runs behind the LOCO columns of `../RESULTS.md`, made with
`evaluation/run_loco.py` on all four training contexts. They have already been run, so there is no
need to run them again to see what the baselines do.

| file | content |
| --- | --- |
| `loco_plain.jsonl` | B1: per held-out context, the LOCO score and every normalised cell-eval metric |
| `loco_gated.jsonl` | B1g: the same for the gated variant |

The final-subset numbers in `../RESULTS.md` come from the verifier; no test predictions are included.
