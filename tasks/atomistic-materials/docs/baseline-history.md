# Earlier uma-s-pretrain runs (maintainers)

Moved out of the agent image from environment/baseline/uma-s-pretrain/RESULTS.md: a stopped v1 pilot,
an unrun starter sample and mechanics probes that predate the current recipe and score.

## From `environment/baseline/uma-s-pretrain/RESULTS.md`

## Earlier runs

| Run | Train / labeled dev | Evaluation | Status |
| --- | ---: | --- | --- |
| Frozen v1 pilot, 30 epochs / 8×H100 | 500,000 / 5,000 | Original 1,200-case three-suite validation; Score v1 | Stopped at ~19.6 epochs before physical validation; predates the v2 exclusions |
| Packaged starter sample | 498,841 / 4,995 | — | Not run end to end |

Earlier probes verified only the mechanics:
- random 146.6M-parameter construction and finite-gradient updates;
- conservative force and stress finite differences;
- export and reload;
- two-step and resumed four-step training on eight H100s.
