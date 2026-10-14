# WN2 results and validation evidence

## Final scientific results

- [Evaluation report](results-summary/README.md): extended 580-update Seed 0,
  original 562-update Seed 0, GenCast and simple references; complete numeric
  tables, curves, compute costs and track limitations.
- [Training curves](training-curves/README.md): original training losses and
  optimizer diagnostics.
- [Extended submission evaluation](extended-evaluation.json) and
  [original submission acceptance](delivery-acceptance-20260912.json).
- [Main training recipe](../WN2_SHORT_TRAIN.md): use the delivered starter's
  schedule; earlier profiling schedules are historical experiments.

## Delivery evidence

- [Delivery validation](../READINESS.md): current supported runtime and limits.
- [Standalone preflight](delivery-preflight.json): clean-checkout and configuration checks.
- [ARCO metadata policy](arco-metadata-policy.json): rolling attributes,
  stable schema checks and historical wind-field spot checks.
- [Standalone setup](../RUNNING.md) and [data preparation](../../environment/data/MATERIALIZE.md).

Other WN2 records preserve implementation probes, hardware measurements and
superseded experiments. They are not active job trackers or additional task
entry points, and they do not supersede the final report. They are excluded
from the starter image. The removed GenCast smoke task and its development
history remain available in Git history.
