# Research record

`/workspace/research` holds the research record next to runs and checkpoints. It must stay
interpretable without the agent transcript. Keep here every config, seed, log, training curve and
checkpoint hash behind a published candidate, plus the two ledgers below.

## `evaluation_history.jsonl`

Append-only, one JSON object per evaluation. Never edit or delete rows; failed and regressed
candidates are evidence. Start it with the baselines' rows from
`/workspace/baseline/evaluation_history.jsonl` (their `verifier` rows come from the isolated
verifier). Required keys:

```
candidate_id, parent_candidate_id, utc_time, kind ("dev_loco" | "verifier"), holdout_context (dev)
or contexts (verifier), scorer ("cellscore.py", cell-eval version), score, normalised {des, pds,
mae, pearson_delta, spearman_lfc, auprc_de, spearman_effect}, raw {...}, submission_sha256,
checkpoint_sha256, train_contexts, seed, hypothesis, status ("ok" | "failed"), failure_reason,
runtime_seconds, config_path, log_path, report_path
```

## `research_report.md`

The living report (template: `/workspace/evaluation/research_report_template.md`). It must always identify the current best
candidate and its snapshot hash, and be updated after every evaluation including failures.
