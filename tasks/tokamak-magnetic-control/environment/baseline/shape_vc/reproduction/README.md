# Reproduction record: shape_vc

| directory | run |
| --- | --- |
| `dev-visible-seeds-0-2-v4/` | `evaluation/harbor_verify.py` defaults (seven cases, visible randomization, seeds 0-2) on `../package/`, 21 workers, `tokamak-shape-metrics/v4`; see `../RESULTS.md` |
| `dev-visible-seeds-0-2/` | the same suite under `tokamak-shape-metrics/v3` (void episodes excluded), kept for history |

Each directory holds the evaluator's `results.json` (every episode: outcome, steady-state errors, policy latency, setup draws such as plasma resistivity, passive scale and supply delay, wall time), `grading.json` (outcome counts per case) and `reward.json` (the flat reward keys). The sealed-verifier tables in `../RESULTS.md` come from verifier runs whose raw outputs are not kept.
