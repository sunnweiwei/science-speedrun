# Submission contract

The submission is a pretrained molecular **encoder**. The verifier never runs your pretraining. It fine-tunes the
encoder with one fixed recipe on each of 24 downstream tasks and scores the test predictions.

## Package

`/workspace/submission/` contains:

| file | content |
|---|---|
| `encoder.json` | `{"arch": "<Uni-Core architecture name>", "user_dir": "<optional sub-directory>", "description": "<optional>"}`. `arch` defaults to `unimol2_84M`. |
| `checkpoint.pt` | `torch.save({"model": state_dict})`; the weights of that architecture. Extra entries such as pretraining heads are ignored. |
| `user_dir/` (optional) | A Python package (`__init__.py`) that registers your own architecture with Uni-Core (`register_model`, `register_model_architecture`). It is imported before the model is built. |
| `README.md` (optional) | Notes. |

No other top-level entries and no symlinks are allowed, and the package is at most 4 GiB. `python
/workspace/baseline/unimol2-ref/publish.py CHECKPOINT` builds such a package from a Uni-Core checkpoint and
publishes it atomically.

## What the model must do

The encoder is fine-tuned by the official Uni-Mol2 fine-tune code: task `mol_finetune`, loss
`finetune_smooth_mae`, Uni-Mol 90f52c4 `unimol2/`. The model built from `arch` must therefore accept that task's
input, `forward(batched_data, classification_head_name=..., features_only=True)` with the Uni-Mol2 molecule
features. Molecules come with explicit 3D coordinates (11 conformers; hydrogens are removed by the pipeline). The
model must also provide `register_classification_head(name, num_classes=...)`. The official `unimol2` model
does all of this; a custom architecture is easiest as a subclass or a re-configuration of it.

## Checks before fine-tuning

`python /workspace/evaluation/check_submission.py` runs the same check as the verifier:

- **Parameter budget.** The model is built as for fine-tuning with one regression head. Its learned parameters,
  excluding the classification heads, must number at most **90,000,000**. The reference `unimol2_84M` has
  83,425,520.
- **Weight coverage.** Every parameter and buffer of that model outside the classification heads must be in
  `checkpoint["model"]` with the same shape. Uni-Core would otherwise load the weights with `strict=False` and
  silently fine-tune a partly random model.

A package that fails either check, or whose code fails, is invalid.

## Fine-tune protocol (fixed, run by the verifier)

For each task, the verifier runs `evaluation/protocol/finetune.sh`, the same code that is in this workspace:

- the official Uni-Mol2 fine-tune recipe: Adam (0.9, 0.99, eps 1e-6), lr 1e-4, polynomial decay with 6 % warmup,
  batch 32 (16 × 2 accumulation), gradient clip 1.0, fp16, pooler dropout 0, targets normalised with train-split
  mean/std;
- **15 epochs**, seed 0, one GPU per fine-tune (up to three fine-tunes share a GPU);
- the epoch with the best validation agg-MAE is kept;
- that checkpoint predicts every test molecule once, averaged over 11 conformers and over duplicate SMILES.

Fine-tune hyperparameters cannot be changed by the package. Each fine-tune has a 3-hour limit and is retried once.
If a task still produces no predictions, the submission is invalid. The fine-tune code runs as an unprivileged
user without network access, and test labels are never in its container.

## Tasks and metrics

24 regression tasks in four groups (`/workspace/data/downstream/manifest.json`):

| group | tasks | train / valid / test molecules | source |
|---|---|---|---|
| QM9 | 6 fine-tunes, 8 properties: HOMO + LUMO + gap (one 3-target task), alpha, cv, mu, r2, zpve | 5,000 / 1,000 / 13,378 | Uni-Mol2 Table 4; DFT geometries; scaffold-similarity split |
| COMPAS-1D | aEA, aIP, dispersion, dipole moment | 5,000 / 870 / 862 | Uni-Mol2; DFT geometries; scaffold-similarity split |
| ExpansionRx | 9 ADMET endpoints (LogD, KSOL, HLM/MLM CLint, Caco-2 Papp A>B and efflux, MPPB, MBPB, MGMB) | 200–4,615 / 22–513 / 209–2,270 | OpenADMET-ExpansionRx 2025–26, official time split |
| ASAP | 5 ADMET endpoints (HLM, KSOL, LogD, MDR1-MDCKII, MLM) | 271–383 / 30–42 / 106–126 | ASAP antiviral-ADMET-2025, official split |

The test sets are complete. The fine-tune training sets are capped at 5,000 molecules, so each fine-tune is short.
ADMET targets are in the challenges' scoring space: log10(clip(y, 0) + 1), with LogD unchanged.

Per task, the verifier reports MAE, RAE = MAE / mean|y − mean(y)|, R², Spearman and Kendall. For
the 3-target QM9 task these are averaged over the targets. A constant prediction gets correlation 0.
Per group, it reports **MA-RAE** (the mean RAE over the group's tasks; this is the ExpansionRx/ASAP
leaderboard metric) and the mean Spearman. MAE, RAE and MA-RAE are better when lower; R², Spearman and
Kendall are better when higher. There is no single aggregate score.
