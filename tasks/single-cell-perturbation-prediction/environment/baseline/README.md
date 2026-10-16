# Baselines

Verified reference implementations. Every baseline is a complete package under the delivery
contract (`bash infer.sh IN OUT`), so the same directory can be checked with
`evaluation/check_submission.py`, scored on the LOCO dev splits with `evaluation/run_loco.py`, and
published to `/workspace/submission` with `publish.py`. Scores below were produced by the isolated
verifier and by `run_loco.py`; each package's `RESULTS.md` has its rows, and
`evaluation_history.jsonl` in this directory holds the raw evaluation records.

| # | Package | Method | Train step |
|---|---|---|---|
| B0 | `noeffect/` | resample basal cells unchanged | none |
| B1 | `meandelta/` with `mode.json = {"gated": false}` | per target, mean of the training contexts' pseudobulk deltas (log1p CP10k), added to resampled basal cells; unknown targets → no effect | `python build_table.py --data /workspace/data --contexts k562 jurkat hct116 hek293t` |
| B1g | `meandelta/` with `mode.json = {"gated": true}` | B1, but no effect if the target gene is not detected in the basal state, and deltas only on genes detected in ≥ 1 % of basal cells | same table |
| B2 | `state_zeroshot/` | STATE: frozen SE-600M + ST trained on the training contexts (`arc-state` @ 9bbfe78) | see its README |

## Reproducing

Work in copies under `/workspace/research`: the B1 table is 1.4 GB and B2's training data about
51 GB. Each package's README has its commands.

```bash
cp -r /workspace/baseline/meandelta /workspace/research/b1 && cd /workspace/research/b1
# B1 / B1g on the four LOCO folds (rebuilds the table without the held-out context each time)
python /workspace/evaluation/run_loco.py --package /workspace/research/b1 \
  --outdir /workspace/research/loco/meandelta \
  --train-cmd "python build_table.py --data {data} --contexts {train_contexts}"
# then build the full table (all four training contexts) and publish atomically:
python build_table.py --data /workspace/data --contexts k562 jurkat hct116 hek293t
python /workspace/baseline/publish.py /workspace/research/b1 \
  --check-input /workspace/data/test/ctx_far/public
```

`publish.py` stages the copy next to `/workspace/submission`, optionally runs
`check_submission.py` on it, and swaps it in with one rename. `mode.json` selects B1 or B1g.

## Scores (release v1, cell-eval 0.8.2; isolated verifier for the test contexts, `run_loco.py` for the folds)

| Baseline | reward (final) | leaderboard | near final | mid final | far final | LOCO k562 | jurkat | hct116 | hek293t |
|---|---|---|---|---|---|---|---|---|---|
| B0 no-effect | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 by construction | 0 | 0 | 0 |
| B1 mean delta | 0.103 | 0.095 | 0.150 | 0.118 | 0.042 | 0.089 | 0.153 | 0.039 | 0.042 |
| B1g gated mean delta | 0.102 | 0.094 | 0.150 | 0.118 | 0.039 | 0.089 | 0.153 | 0.026 | 0.028 |
| B2 STATE zero-shot (gene-space ST, 20k steps) | 0.046 | 0.034 | 0.063 | 0.033 | 0.043 | 0.011 | 0.063 | – | – |

Per-metric normalised scores on the final subsets (same runs; each column is
`(m - no_effect) / (1 - no_effect)`, MAE is `1 - m / no_effect`, clipped at 0, and the context
score is their mean):

| Baseline | context | DES | PDS | MAE | pearson_delta | spearman_lfc | auprc_de | spearman_effect |
|---|---|---|---|---|---|---|---|---|
| B1 | near | 0.069 | 0.097 | 0.006 | 0.147 | 0.249 | 0.000 | 0.480 |
| B1 | mid | 0.079 | 0.037 | 0.002 | 0.128 | 0.219 | 0.000 | 0.359 |
| B1 | far | 0.063 | 0.024 | 0.000 | 0.074 | 0.132 | 0.000 | 0.000 |
| B1g | near | 0.069 | 0.097 | 0.006 | 0.147 | 0.249 | 0.000 | 0.480 |
| B1g | mid | 0.079 | 0.037 | 0.002 | 0.128 | 0.219 | 0.000 | 0.359 |
| B1g | far | 0.069 | 0.024 | 0.000 | 0.075 | 0.100 | 0.007 | 0.000 |
| B2 | near | 0.061 | 0.006 | 0.000 | 0.010 | 0.027 | 0.009 | 0.325 |
| B2 | mid | 0.085 | 0.006 | 0.000 | 0.000 | 0.000 | 0.002 | 0.138 |
| B2 | far | 0.188 | 0.000 | 0.000 | 0.038 | 0.011 | 0.064 | 0.000 |

B0 is 0 in every cell by construction (it *is* the normalisation baseline). The rows come from
the `verifier` entries of `evaluation_history.jsonl` (this directory), which also carry the raw
(un-normalised) values and the per-stratum breakdowns.

## No frontier reference point yet

There is no published number for a fully trained frontier model on *this* evaluation, and none
of the rows above should be read as one. B2 uses the SOTA architecture (Arc's STATE) but at this
task's dev budget: `state_sm`, 20k steps, a 7,330-gene common panel, about 6 GPU-hours. Two ways
to fill the gap, neither done yet: (a) run Arc's released STATE checkpoint zero-shot on the three
test contexts — the obstacle is that its SE-space output needs `FinetuneVCICountsDecoder`, whose
paths are hard-coded, to come back to gene space; (b) calibrate against the real VCC 2026
validation leaderboard by submitting one of these baselines, which tests whether this proxy score
orders candidates the same way the official one does. The VCC 2025 leaderboard is not directly
comparable: ctx_far is built from that dataset, but the metric panel and the normalisation
differ.

Reading the table: the near > mid > far ordering is the context-distance gradient the task is
built to expose; the far context (Flex chemistry) keeps only ~30 % of the transfer
that works on the near context.
A group score is the mean of the six per-perturbation metrics (the effect-size
Spearman is a whole-run property and is not split by group); the group numbers quoted here are
computed from their per-metric values. For B1 the withheld targets score 0 in every context because the table
has nothing to say about them, and `conserved-strong` targets transfer better than
`context-variable` ones (all 300 requests: 0.129 vs 0.057 near, 0.106 vs 0.060 mid, 0.051 vs
0.044 far) — the advantage shrinks with context distance.
Gating deltas by basal detection (B1g) changes nothing on the 3' v3 panels (they contain only
expressed genes) and slightly hurts on the far context and the Orion folds — the naive gate is
not the right answer to Q1. B2 (STATE) beats B0 everywhere but loses to the delta table
overall (0.046 vs 0.103) — the same pattern the 2026 benchmark papers report — with two
instructive exceptions on the far context: its DES is 0.19 vs 0.07 for B1, and it scores 0.12 on
the withheld targets where B1 scores 0 (ESM-2 perturbation features generalise to genes never
perturbed in training), while it gives up PDS and MAE. B2 folds hct116/hek293t were not run
(GPU time); each B2 training is 20k steps ≈ 1.5 h on an uncontended H100 (4.5 h under load).
Rows are reproducible from this directory; runtimes: table build < 2 min, B1 inference 8–30 s
per context, B2 inference 2–5 min per context.
