# Evaluation and analysis utilities

The current baseline measurements are documented only in
[baseline/starter/RESULTS.md](../../environment/baseline/starter/RESULTS.md).

`screening.py` prepares development-only data views and supports cached-feature
validation. `compare_seeds.py` computes paired comparisons and uncertainty from
locally generated predictions. Both are copied into the agent evaluation workspace.
Other scripts are optional analysis utilities that consume locally produced data;
their presence does not establish a completed experiment or a track baseline.

Retain generated predictions, manifests and runtime records privately when
reproducing experiments. Do not publish scheduler identities or deployment paths.
