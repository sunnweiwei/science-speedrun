# Zero-shot virtual cell: predict CRISPRi responses in cellular contexts you have never seen perturbed

## Task and goal

You are given the **unperturbed (non-targeting) single cells of a cellular context** and a list of
**CRISPRi target genes**. The context's identity is hidden: no cell-line name, no tissue, only its
basal expression and its chemistry. Predict the single-cell transcriptomic response to each
knockdown in that context. No perturbation data from that context exists anywhere in your
workspace. This is the setting of the Arc Institute Virtual Cell Challenge 2026. Build, train,
evaluate and keep improving one model for this problem.

Same-context perturbation prediction (VCC 2025) was essentially solved by statistics: given a
context's own training perturbations, cross-dataset mean effects plus rank calibration matched or
beat every deep model. Under **context shift**, the 2026 benchmark papers find that deep models
regress to those same baselines. What no method does well yet is to infer, from the basal state
alone, *what a knockdown will do here*. Three sub-questions make that concrete, and each is a
separately reported stratum of the score:

| | Question | Stratum in the report |
|---|---|---|
| Q1 | Does knocking down *g* have an effect in this context, and how strong? (basal expression of *g*, dependency, pathway state) | targets binned by their effect pattern across the training contexts: `conserved-strong`, `context-variable`, `weak` |
| Q2 | For responsive targets, which part of the response is context-invariant (core complexes) and which part is rewired? | per-stratum DES / PDS; the operator report adds conserved vs. context-specific DE gene sets |
| Q3 | Targets never perturbed in any training context | `seen` vs. `withheld` targets |

Q1 decides most of the score (DES and PDS are both gated on predicting an effect at all), Q2 is
where a scientific result would come from, and Q3 is the long-run capability.

The score of a test context is the mean of seven `cell-eval` 0.8.2 Generalist metrics (DES, PDS,
MAE, Pearson-delta, Spearman of significant-gene LFC, AUPRC of DE recovery, Spearman of effect
size), each normalised against the no-effect baseline and clipped at 0. The reward is the mean over
the three test contexts of the final-subset score (200 of each context's 300 requests); the other
100 form a separately reported leaderboard subset. Details: `/workspace/contract.md`.

| baseline | reward | near | mid | far |
|---|---:|---:|---:|---:|
| B0 no effect | 0.000 | 0.000 | 0.000 | 0.000 |
| B1 cross-context mean delta | 0.103 | 0.150 | 0.118 | 0.042 |
| B1g basal-gated mean delta | 0.102 | 0.150 | 0.118 | 0.039 |
| B2 STATE zero-shot (SE-600M + ST, 20k steps) | 0.046 | 0.063 | 0.033 | 0.043 |

Aim to beat B1 on every context, and write the research report so that a reader learns what your
model knows about Q1–Q3.

## Workspace

- `/workspace/contract.md`: the package, the `infer.sh` inputs and outputs, the inference limits
  and the scoring.
- `/workspace/data/` (read-only; nothing else exists):
  - `README.md` documents schemas, provenance, licences, sampling caps and QC;
  - `train/<ctx>/` for the training contexts **K562** (Replogle 2022), **Jurkat** (Nadig 2025),
    **HCT116** and **HEK293T** (X-Atlas/Orion), each with full perturbation data: `cells.h5ad`
    (raw counts; a seeded per-target sample of cells), `pseudobulk.h5ad` (per-target means, deltas
    and z-scores over **all** source cells), `se_embeddings.npy` (SE-600M embeddings of every
    shipped cell) and `dev_split/`, a test-shaped view for leave-one-context-out development;
  - `test/<ctx>/public/` for `ctx_near`, `ctx_mid` and `ctx_far`: basal cells, 300 requests and
    `context.json`. Their identities are hidden; `ctx_far` is on 10x Flex chemistry (the VCC 2026
    assay), the others on 10x 3' v3. Their responses exist only in the verifier. A set of 80 target
    genes is **withheld from every training context** and appears in the test requests; you are
    not told which;
  - `priors/`: ESM-2 gene embeddings, GO annotations, and DepMap gene-effect scores for the
    training lines in DepMap;
  - `se600m/`: the frozen SE-600M release, also at `/opt/se600m`.
- `/workspace/baseline/`: verified packages with reproduced scores: `noeffect/` (B0),
  `meandelta/` (B1, B1g), `state_zeroshot/` (B2), and `publish.py` for atomic publication.
- `/workspace/references/`: the installed `arc-state` and `cell-eval` sources and related work,
  not verified here.
- `/workspace/evaluation/`: `cellscore.py` (the verifier's scorer), `run_loco.py`,
  `check_submission.py`, and the research-record template.
- `/workspace/research/`: writable space for runs, checkpoints and the research record.

## Rules

- **Closed world.** Only the mounted data and weights may be used; they are all listed in
  `data/manifest.json` with provenance and licence. There is no network; do not try to reach one.
- **One generic model.** `infer.sh` implements `basal + targets -> predictions` for arbitrary
  contexts, gene panels and target lists. Per-context lookups keyed on the shipped test inputs,
  hard-coded test target lists, cached predictions and any attempt to recover the private responses
  are forbidden.
- Everything the package needs at inference must be in `/workspace/submission` or at `/opt/se600m`.
- SE-600M is a frozen cell-state encoder whose embeddings are precomputed; treat it as a fixed
  representation unless you have a measured reason not to (raw counts are shipped too). The
  research object is everything between the basal cells and the prediction: how a perturbation is
  represented (learned, ESM-2, co-essentiality, GO), **how the context gates it**, the training
  objective (pseudobulk regression, distribution matching, objectives aimed at DES/PDS), and the
  cross-context data strategy. Model scale is yours to choose.

## Submission

Keep a complete package at `/workspace/submission` from early on (`/workspace/contract.md`): the
inference code, configuration, all learned weights, preprocessing and a concise `README.md`.
Check every candidate with `evaluation/check_submission.py` before publishing it. The package may
be read at any moment, so replace it atomically, with `/workspace/baseline/publish.py` or the same
stage-and-rename pattern. Keep training data views, optimizer state and old candidates under
`/workspace/research`.

Keep the research record in `/workspace/research` (`/workspace/evaluation/RESEARCH_RECORD.md`):

- `evaluation_history.jsonl`: one row per evaluation, appended, never edited;
- `research_report.md`, from the template: the current best candidate and its snapshot hash;
  model, data, training and reproducibility; the chronological candidate table; comparison with
  B0/B1/B1g/B2 on every fold; per-stratum analysis (Q1–Q3); ablations; negative results; compute
  use; limitations; and a ranked list of next hypotheses. Update it after every evaluation,
  including failures;
- configs, seeds, logs, training curves and checkpoint hashes for every published candidate.

## Continuous research

1. **The baselines are already reproduced.** `baseline/*/RESULTS.md` has the verifier and LOCO
   numbers of B0, B1, B1g and B2 (STATE zero-shot), and `baseline/*/reproduction/` the per-metric
   LOCO scores and STATE's training logs. Do not rerun them to confirm the numbers. Check your own
   train -> `infer.sh` -> score path once on one LOCO fold with a short run.
2. **Leave-one-context-out is your zero-shot signal.** Train on three training contexts and score
   the fourth's `dev_split/` with the verifier's code; report all four folds. For a Q3 signal,
   hold out your own target set from the training contexts.
3. Then loop without end: measure, inspect per-target failures by stratum, state a falsifiable
   hypothesis, make the smallest change that tests it, train, score on the unchanged LOCO folds,
   publish if better, record the result and the next hypothesis, and start the next experiment.

Do not return a final answer or declare the task complete. A reproduced baseline, a new best score
or an exhausted idea list is a milestone, not the end of the task.
