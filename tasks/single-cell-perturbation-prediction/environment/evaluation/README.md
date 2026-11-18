# Development evaluation

V3/V4 uses `population_metrics_v2.py` and `calibrated_reward_v2.py` (`score_context_v3`) to score
held-out training contexts using the exact verifier formula (`max(0, 0.5 * (amplitude_skill +
identification_skill))`), and `query_independence.py` in `check_submission.py` to verify two-pass
per-request query independence (`atol=1e-4, rtol=1e-4`). `cellscore.py` computes the historical
cell-eval 0.8.2 diagnostics.

| Script | Use |
|---|---|
| `check_submission.py --package DIR --input DIR` | contract check: runs `infer.sh` from a relocated copy in a scratch directory and validates `predictions.h5ad` exactly as the verifier does (gene order, cell counts, non-negative finite values) |
| `run_loco.py --package DIR --outdir DIR [--train-cmd CMD]` | leave-one-context-out: for each training context, optionally retrain on the other three, run `infer.sh` on its `dev_split/` and measure v2 calibrated WMSE skill plus the v1 diagnostics; writes `OUT/<ctx>/report.json` and `OUT/summary.json` |
| `cellscore.py --predictions P --basal B --responses R --requests Q --outdir O` | score one context: the seven normalised Generalist metrics and their mean (see `/workspace/contract.md`) |

The test contexts' responses are not in this container, so `dev_split/` of the four training
contexts is the development signal. Its no-effect tables are cached under
`dev_split/noeffect/`, so `run_loco.py` scores only the candidate side. Write every output under
`/workspace/research`.

## Research record

Keep these under `/workspace/research` (details in [RESEARCH_RECORD.md](RESEARCH_RECORD.md)):

- `evaluation_history.jsonl`: one row per evaluation, appended, never edited. The baselines' rows
  are in `/workspace/baseline/evaluation_history.jsonl`; copy them in as the first rows.
- `research_report.md`: the living report, started from
  [research_report_template.md](research_report_template.md).
