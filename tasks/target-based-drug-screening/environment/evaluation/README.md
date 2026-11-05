# Development evaluation

The reference computes BioLiP development retrieval metrics after every
fine-tuning epoch and selects the checkpoint by development BEDROC(alpha=80.5).
To evaluate an already exported dual Uni-Mol package without retraining:

```bash
python /workspace/evaluation/dev.py /workspace/research/candidate \
  /workspace/research/dev-metrics.json
```

For a short CPU interface check, append
`--device cpu --max-records 8 --validation-batch-size 4`. Remove the record
limit for a full development evaluation. Keep the batch size, data, crop seed,
and number of processes fixed when comparing models: these determine the
in-batch retrieval candidates. This command uses one process and epoch-zero
preprocessing, as in single-process training. Its score is not directly
comparable to a development run with different batch construction, nor is it
the held-out screening EF1% score. The output records these conditions.

`metrics.py` contains the same pure ranking-metric function imported by the
independent screening scorer: ROC-AUC, BEDROC(alpha=80.5), and enrichment
factors at 0.5%, 1%, 2%, and 5%. `ranking_metrics(values)` expects finite
`[score, binary_label]` pairs sorted by descending score and both classes.
Exactly tied scores are averaged over all possible within-tie orders.
Use only supplied dev records or synthetic examples for local experiments.

The agent image excludes the held-out label reader, input-permutation builder,
and verifier. In the source repository these controller-only entry points
remain `score.py` and `input_view.py`; neither is needed for dev evaluation.

`aggregation.py` contains the same public aggregation used by the verifier.
The primary protocol is `drugclip-dataset-bedroc/v2`: take mean BEDROC(80.5)
within each dataset, then average DUD-E and LIT-PCBA equally. A primary score
requires all 102+15 targets. Partial dev examples can produce diagnostics but
not a full-benchmark reward. No hidden data or labels are supplied here.
