# Single-Cell Perturbation Prediction

## 1. Overview

Develop one model that predicts single-cell CRISPRi knockdown responses in a cellular context it
has never seen perturbed, given only that context's unperturbed cells and the target list. This is
the Arc Virtual Cell Challenge 2026 zero-shot setting, built from public data so that it can be
scored locally today and swapped to the real VCC 2026 lines when their labels are released.

Four training contexts have full perturbation data (K562, Jurkat, HCT116, HEK293T). Three
anonymised test contexts are tiered by distance from the training corpus: a near cancer line, a mid
non-cancer line, and a far pluripotent line on the 10x Flex assay VCC 2026 uses. 80 target genes
are withheld from every training context so that never-seen targets are measurable. Each test
context has 300 requests, split into a leaderboard subset (100) and a final subset (200).

- **Reward:** the mean over the three test contexts of the final-subset score; a context score is
  the pinned `cell-eval` 0.8.2 Generalist panel normalised against the no-effect baseline.
  Strata (effect pattern across training contexts, seen vs. withheld targets) are reported
  separately, so the three scientific sub-questions (does a knockdown act here, which part of the
  response transfers, what about unseen genes) are each visible.
- **Status:** verified. B1 (cross-context mean delta) scores 0.103, B2 (STATE zero-shot) 0.046.

## 2. Data

| Source | Role | License |
|---|---|---|
| Replogle 2022 genome-wide + essential K562 (Figshare 20029387) | training | CC BY 4.0 |
| Nadig 2025 Jurkat (GEO GSE264667) | training | GEO terms |
| X-Atlas/Orion HCT116, HEK293T (`Xaira-Therapeutics/X-Atlas-Orion` @ `53a5bc98`) | training | CC BY-NC-SA 4.0 |
| three public CRISPRi screens, identities withheld (one from the Arc VCC 2025 objects) | test | per source |
| DepMap 24Q4, GO, ESM-2 gene embeddings | priors | CC BY 4.0 / per source |
| SE-600M (`arcinstitute/SE-600M` @ `5a9a80f`) | frozen encoder | Arc model licence |

The prepared release v1 is on Hugging Face, `sunweiwei/ai4sci-virtual-cell-zeroshot-data` (public;
revision pinned in `environment/data/download-source.json`): `dev/` about 169 GB plus SE-600M
12 GB, `private/` about 22 GB. Training contexts after the per-target cell caps: 2,568,180 cells,
167.5 GB. [`environment/data/MATERIALIZE.md`](environment/data/MATERIALIZE.md) has the fetch and
rebuild procedures:

```bash
bash single-cell-perturbation-prediction/environment/data/fetch_release.sh DATA_ROOT
export TASK_DATA_DIR=DATA_ROOT/dev TASK_PRIVATE_DIR=DATA_ROOT/private
```

```text
DATA_ROOT/
  manifest.json   operator manifest (test-context identities; never mounted)
  dev/            train/ test/<ctx>/public/ priors/ se600m/ manifest.json splits.json gene_panel*.{csv,json}
  private/        test responses, request subsets and strata, cached no-effect and DE tables (grader only)
```

Schemas, sampling caps, QC and per-context sizes: [`environment/data/README.md`](environment/data/README.md).

## 3. Workspace

```text
/workspace/
  instruction.md, contract.md
  data/README.md + read-only mounts of dev/: train/ test/ priors/ se600m/ manifest.json splits.json gene_panel*
  baseline/        noeffect/ (B0), meandelta/ (B1, B1g), state_zeroshot/ (B2), publish.py, evaluation_history.jsonl
  references/      installed arc-state / cell-eval, related work
  evaluation/      cellscore.py (= the verifier's scorer), run_loco.py, check_submission.py, research-record template
  submission/      empty at start
  research/        empty, writable, large disk
/opt/se600m -> /workspace/data/se600m
```

The entrypoint refuses to start if a private tree is visible. Baselines, each a complete package
with README and reproduced RESULTS:

| # | Package | Method | Train step |
|---|---|---|---|
| B0 | `noeffect/` | resample basal cells unchanged | none |
| B1 | `meandelta/` with `mode.json = {"gated": false}` | per target, mean of the training contexts' pseudobulk deltas, added to resampled basal cells | table build < 2 min |
| B1g | `meandelta/` with `mode.json = {"gated": true}` | B1, gated by basal detection of the target and of each gene | same table |
| B2 | `state_zeroshot/` | STATE: frozen SE-600M + ST (`state_sm`, gene space, ESM-2 perturbation features) trained on the training contexts | 20k steps, about 1.5 h on one H100 |

## 4. Evaluation

[`tests/README.md`](tests/README.md). Two containers from `tests/Dockerfile`, no network:

- `main` mounts only the public test inputs and SE-600M, copies the package, runs `infer.sh` once
  per context as uid 10001 on one GPU (60 minutes per context), validates each
  `predictions.h5ad`, and seals the predictions with their hashes;
- `grader` mounts the public inputs and the private responses, checks provenance, and scores every
  context with `cellscore.py` on the leaderboard and final subsets and by stratum.

`reward.json` has `reward` (mean final-subset score), `leaderboard_reward` and per-context values.
A context whose inference fails scores 0 without hiding the others; scorer failures leave no
reward. `agent_feedback.json` (leaderboard subset only) exists for runners that return feedback;
under TASK_STANDARD.md nothing is returned to the agent.

## 5. Running

```bash
export TASK_DATA_DIR=DATA_ROOT/dev TASK_PRIVATE_DIR=DATA_ROOT/private   # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/vcz-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p single-cell-perturbation-prediction -a <agent> -m <model> --allow-agent-host <model API host>
```

**Resources (external budget, in `task.toml`).** Development: 8× H100, 48 hours, at least 64 CPUs, 512 GiB memory,
1 TiB storage; the release is mounted, not copied. Verifier: one GPU, at most 60 minutes per
context, about 25 minutes to score three contexts.

Unit tests (need `cell-eval`, so they run in the verifier image):
`bash single-cell-perturbation-prediction/tests/run_unit_tests.sh vcz-verifier:latest`.

## 6. Results and validation

Release v1, cell-eval 0.8.2; isolated verifier for the test contexts, `run_loco.py` for the folds:

| Baseline | reward (final) | leaderboard | near | mid | far | LOCO k562 | jurkat | hct116 | hek293t |
|---|---|---|---|---|---|---|---|---|---|
| B0 no effect | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| B1 mean delta | **0.103** | 0.095 | 0.150 | 0.118 | 0.042 | 0.089 | 0.153 | 0.039 | 0.042 |
| B1g gated mean delta | 0.102 | 0.094 | 0.150 | 0.118 | 0.039 | 0.089 | 0.153 | 0.026 | 0.028 |
| B2 STATE zero-shot | 0.046 | 0.034 | 0.063 | 0.033 | 0.043 | 0.011 | 0.063 | – | – |

The near > mid > far ordering is the context-distance gradient the task is built to expose. B2
beats B0 everywhere but loses to the delta table overall, the pattern the 2026 benchmark papers
report; on the far context it has the best DES and scores 0.12 on the withheld targets where B1
scores 0. Per-metric tables: `environment/baseline/*/RESULTS.md`; raw records:
`environment/baseline/evaluation_history.jsonl`. The developer's verifier runs were cross-checked
by the task reviewers, not re-run.

## 7. Limitations

- There is no frontier reference point: B2 is the SOTA architecture at a small budget, and Arc's
  released STATE checkpoint was not run zero-shot on these contexts.
- External calibration is pending: no baseline has been submitted to the VCC 2026 validation
  leaderboard, so it is untested whether this proxy orders candidates like the official score.
- Raw cells are subsampled per target for the training contexts (pseudobulk tables are complete);
  K562 merges two screens.
- The near/mid test contexts have about 45–70 cells per target in the source data, so their
  per-target DE is noisier than the far context's.
- All test inputs are public upstream; the hidden identities and sealed responses make this a
  benchmark-visible evaluation, not a secret test set.

## 8. Citation and licenses

Task author: sunnweiwei. Upstream: STATE (Adduri et al., bioRxiv 2025) and `arc-state`
(`9bbfe78`); `cell-eval` 0.8.2 (Arc Institute); Replogle et al., Cell 2022; Nadig et al., Nature
Genetics 2025; X-Atlas/Orion (Xaira Therapeutics); DepMap 24Q4; Gene Ontology; ESM-2; SE-600M.
Each dataset and model keeps its own license (section 2); X-Atlas/Orion is non-commercial.
