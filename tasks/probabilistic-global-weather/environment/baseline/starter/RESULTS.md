# Native WN2 starter: results

Weather reward = `1/(1 + normalized CRPS)` on the eight-date 2022 panel (56 members, 30 leads
through 15 days, native 0.25° grid); larger is better.

| run | who, where | result |
| --- | --- | --- |
| 580-update recipe, seed 0 | task author, Apptainer runtime built from the task Dockerfiles | independent evaluation reward **0.680651** (normalized CRPS 0.469183); 2.20 % lower normalized CRPS than the original 562-update seed 0 |
| 580-update recipe, seed 0 | task reviewers, Docker, four H100 80 GB | retrained from scratch: all eleven stages passed in 13.1 h, 183.8M-parameter package; full verifier in Docker (5.1 h) reward **0.674377** (normalized CRPS 0.482850). Record: `reproduction/` |

Comparisons of the author's run with GenCast, the original seed and simple references on the full
test subset are in `reproduction/test-evaluation/` (summary tables and the per-variable, level and lead
CSV), copied from `docs/validation/results-summary/` in the task repository.
