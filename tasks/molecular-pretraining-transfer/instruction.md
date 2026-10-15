# Pretrain a molecular foundation model that transfers to quantum-chemistry and ADMET tasks

## Task and goal

Pretrain a molecular encoder from scratch on a fixed corpus of 3D molecules. The encoder should transfer well
after short fine-tuning to many property-prediction tasks. Uni-Mol2 (2024) is the reference: a two-track
atom/pair transformer pretrained by masked-atom and coordinate/distance denoising on 3D conformers. Its official
code, recipe and 84M architecture are the starting point here. You may change anything about pretraining: the data
pipeline, objectives, architecture (within the parameter budget), schedule and optimisation.

The encoder is judged on 24 regression tasks after a fixed, short fine-tune per task (15 epochs, at most 5,000
training molecules):

- **QM9**: 8 quantum-chemical properties;
- **COMPAS-1D**: 4 properties of polycyclic aromatic systems;
- **OpenADMET-ExpansionRx**: 9 ADMET endpoints from a 2025 drug-discovery campaign, time split;
- **ASAP antiviral-ADMET-2025**: 5 ADMET endpoints.

For every task the verifier reports MAE, RAE, R², Spearman and Kendall on the complete test set. For every group it
reports MA-RAE (mean RAE, lower is better) and mean Spearman. The details are in `/workspace/contract.md`. Current
test MA-RAE:

| encoder | QM9 | COMPAS-1D | ExpansionRx | ASAP |
|---|---:|---:|---:|---:|
| no pretraining (same architecture, random init) | 0.413 | 0.662 | 1.078 | 1.315 |
| baseline: Uni-Mol2 84M, paper recipe, 30k updates | 0.183 | 0.166 | 0.744 | 0.636 |
| released Uni-Mol2 84M (full-length pretraining; not available here) | 0.192 | 0.121 | 0.668 | 0.644 |

Aim to improve every group over the baseline, and to reach or pass the released model.

## Workspace

- `/workspace/contract.md`: the package format, the checks, the fixed fine-tune protocol, the tasks and the
  metrics.
- `/workspace/data/` (read-only):
  - `README.md` describes the data;
  - `pretrain/`: the Uni-Mol2 pretraining corpus, shards 0–11 (about 126 M molecules with 3D conformers), with every
    evaluation molecule removed; `valid_part_0.lmdb` is a 100k-molecule validation set;
  - `downstream/`: the fine-tune training and validation splits of the 24 tasks.
- `/workspace/baseline/unimol2-ref/`: the verified reference, with `pretrain.sh`, `publish.py`, a README and
  RESULTS. It is the official Uni-Mol2 code and paper recipe on this corpus, with a shortened schedule.
- `/workspace/references/`: notes on Uni-Mol2, the released models and an earlier small-scale study, not verified
  here.
- `/workspace/evaluation/`:
  - `check_submission.py`: the verifier's package check;
  - `dev_eval.py`: the verifier's fine-tune protocol on the development splits;
  - `protocol/`: the verifier's own fine-tune code.
- `/workspace/submission/`: your current package; empty at start.
- `/workspace/research/`: writable space for data, checkpoints, logs and notes.

## Rules

- Pretrain from scratch. Use `/workspace/data/pretrain` and anything you compute from it (for example new
  conformers or RDKit-derived features). No other pretrained weights or external data may be used.
- The downstream data are for evaluating candidates, not for training the encoder. Do not use their molecules or
  labels in pretraining. The verifier does all fine-tuning itself, with its fixed recipe.
- The encoder may have at most 90 M parameters (see the contract). Test molecules and labels are not in this
  container; do not try to obtain them.
- There is no network access.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission` from early on, and replace it
whenever you have a better encoder. The package may be read at any moment, so replace it atomically:
`python /workspace/baseline/unimol2-ref/publish.py CHECKPOINT` stages the package next to the target, checks it and
swaps it in with one rename. Start with the baseline:

```bash
bash /workspace/baseline/unimol2-ref/pretrain.sh     # checkpoints in /workspace/research/unimol2-ref/ckpt
python /workspace/baseline/unimol2-ref/publish.py /workspace/research/unimol2-ref/ckpt/checkpoint_last.pt
python /workspace/evaluation/dev_eval.py --encoder /workspace/submission --out /workspace/research/eval/baseline
```

An intermediate checkpoint can be published while pretraining is still running.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

There is no end point. Measure candidates with `dev_eval.py` on all four groups, find where transfer is weakest,
form a hypothesis about pretraining, test it, and publish when it is better. Then start the next experiment. Keep
concise notes in `/workspace/research`. Do not stop to hand in a final answer; a published encoder or a reached
score is a milestone, not the end of the task.
