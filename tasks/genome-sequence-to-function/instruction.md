# Develop a human sequence-to-function model at AlphaGenome scale

## Task and goal

Design, train and keep improving one model that reads 1,048,576 bp of human DNA and predicts its
functional genomics readout across eleven output families and 5,930 tracks:
- RNA-seq, CAGE, PRO-cap, ATAC, DNase, histone ChIP-seq and TF ChIP-seq;
- splice sites, splice-site usage and splice junctions;
- 3D contact maps.

This is the AlphaGenome problem, on AlphaGenome's own public training data and fold split
(`FOLD_0`). Training uses every eligible interval of the training folds.

The same model is judged on what such models are for: predicting the effect of genetic variation.
- **Track prediction** (the reward): per-track Pearson on 1,000 held-out test intervals, averaged
  within each family and then equally across families.
- **Variant and perturbation effects** (reported next to the reward, with published anchors):
  expression QTL sign, causality and effect size; splicing QTLs; ClinVar non-coding, splice-region
  and missense variants; MFASS splicing variants; chromatin-accessibility and SPI1-binding QTLs;
  CRISPRi enhancer–gene links; and TraitGym causal variants for Mendelian diseases and complex
  traits. Each comes with AlphaGenome's, or Borzoi's, Enformer's and CADD's, score on the same
  variants.

Track Pearson is where most models do reasonably well. The variant evaluations are where current
models, including much larger ones, are still far from solved. For example, on TraitGym's Mendelian
variants, sequence-to-function models trail conservation-based scores by a wide margin. Progress
there is the scientifically interesting outcome.

## Workspace

- `/workspace/contract.md`: the two entry points (`infer.sh`, `perturb.sh`), every output shape,
  and how each is scored.
- `/workspace/data/` (read-only):
  - `README.md` and the reference genome;
  - public manifests for train and validation;
  - train and validation targets, labels and scoring queries;
  - `variant_evals/valid/`, a development split of every variant evaluation, with labels.

  Train targets are stored as 8-bit codes with per-track ranges. The supplied dataset reader
  decodes them per example; validation targets are full precision.
- `/workspace/baseline/alphagenome-reference/`: the verified baseline, an AlphaGenome-style
  reference model with its complete code:
  - the model, data loader and multi-GPU training (`reference/alphagenome_50m/`, `run_ddp.sh`);
  - `predict.py`/`infer.sh` and `perturb.py`/`perturb.sh`, which write the required outputs;
  - `package.sh` and `publish.sh`, a README with the commands, and RESULTS with the scores we
    reproduced (215M parameters, trained on all training intervals).
- `/workspace/references/`: related models and papers, not verified in this setup.
- `/workspace/evaluation/`: the verifier's own scoring code. `score.py` scores validation track
  predictions, `score_variants.py` scores the development variant split, and `variant_scoring.py`
  defines every variant metric.
- `/workspace/submission/`: the current package; empty at start.
- `/workspace/research/`: writable space for experiments, checkpoints and notes.

## Rules

- **One model serves everything**: every track, both entry points and every evaluation. An
  ensemble counts as one model.
- **Model size is not capped; the training budget is the limit.** Choose architecture and scale for
  what you can train well in the time and hardware you have.
- `FOLD_0` is fixed: train on the training folds, select on validation, never touch test targets.
  Do not use test labels, label-derived caches, the variant labels behind the public perturbation
  queries, or any pretrained AlphaGenome (or other model's) parameters trained on these folds.
- Inference is offline and uses only the public inputs: DNA from the pinned reference and the
  public manifests. There are no splice coordinates or targets at inference.
- There is no network access. Vendor every dependency and asset into the package.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission` from
early on. It must contain:
- executable `infer.sh INPUT_ROOT OUTPUT_ROOT` and `perturb.sh INPUT_ROOT OUTPUT_ROOT`;
- the model code and weights;
- a `README.md` covering the architecture, parameter count, training provenance and compute,
  validation results, and limitations.

Replace the package atomically: build it elsewhere, check it with `score.py` on validation, then
swap it in, as the baseline's `publish.sh` does. Keep checkpoints and experiments under `/workspace/research`.

## Continuous research

There is no end point. The baseline has already been trained and scored in this setup:
`baseline/alphagenome-reference/RESULTS.md` has its numbers and `reproduction/` its training log,
per-track validation Pearson and development variant scores. Do not train it again to confirm them.
Check your own pipeline with a short run instead: a few training steps, `score.py` on validation,
`score_variants.py` on the development variant split, and publication. Then repeat:

1. Find where the model is weakest, per family and per variant evaluation.
2. Change one thing: architecture, scale, objective, data use or schedule.
3. Measure on validation (tracks and the development variant split), and publish the package
   whenever it improves.

Keep a concise experiment ledger in `/workspace/research`. Do not stop to hand in a final answer; a
published model is a milestone, not the end of the task.
