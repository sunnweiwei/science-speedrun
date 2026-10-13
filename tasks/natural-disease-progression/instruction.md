# Develop a neural model that anticipates serious new disease after discharge

## Task and goal

Design, implement, train and keep improving one neural network that reads a patient's multimodal
history up to a hospital discharge and predicts their risk of future first-recorded disease and
death. You may study and modify the supplied baseline and references, but you are responsible for
proposing, testing and selecting the model structure.

Each sample is one patient at their second eligible live adult discharge. For every sample,
predict cumulative risk at `[30, 60, 90, ..., 360, 365]` days for every disease in the frozen
vocabulary and for death (`/workspace/contract.md`). The endpoint is a first recorded inpatient
diagnosis in MIMIC-IV, not biological onset.

**The score is case detection for serious new disease.** It uses a prespecified panel of serious,
actionable outcomes: heart failure, atrial fibrillation, myocardial infarction, stroke, pulmonary
embolism, venous thrombosis, aortic disease, type 2 diabetes, chronic kidney disease, acute kidney
injury, sepsis, pneumonia, respiratory failure, cirrhosis, hepatic failure, dementia, COPD,
interstitial lung disease, and several cancers and metastases. For each endpoint, patients whose
history already records the condition or a direct precursor are excluded. The remaining patients
are ranked by your 365-day risk, and the score is the share of next-year cases found among the top
5 %: if a hospital can follow up only 1 in 20 discharged patients, how many future cases does your
model reach? Chance is 0.05 and perfect is 1.0. The reward averages this over the panel endpoints
with enough cases; `/workspace/contract.md` and `/workspace/evaluation/panel_scoring.py` define it
exactly. Disease-first macro AUROC over all diseases, AUPRC, calibration and mortality are reported
next to it.

Validation and test patients are all discharged after the October 2015 switch from ICD-9 to
ICD-10, so their outcomes are completely coded. Training also contains patients discharged
earlier, and their one-year outcome labels are incomplete: many ICD-9-era records did not map to a
category and are absent rather than negative. Deciding how to use those patients is part of the
problem.

This is a **neural architecture development track**. A compliant final model:

- learns a patient representation through gradient optimization on the training partition and
  jointly predicts all diseases;
- may use modality-specific encoders, disease-specific output heads, learnable temporal
  components, or internal neural experts;
- produces patient-specific risks for all 13 horizons with a learnable multi-time head,
  time-conditioned decoder, cumulative hazard, or another learned temporal mechanism; and
- does not obtain its final predictions from LightGBM, random forests, stratified rate tables,
  fixed population time scaling, or an ensemble containing such predictors.

Non-neural methods are permitted only as private sanity checks. A gradient-boosted tree on
hand-built features is a strong reference on this data (see the baseline's RESULTS.md); beating
it with a learned representation is the point. Re-running the baseline unchanged, or changing
only a seed, epoch count, learning rate or batch size, is not architecture development. Explain a
concrete structural change and the validation evidence used to keep or reject it.

## Workspace

- `/workspace/contract.md`: the submission package, prediction format and the exact score.
- `/workspace/data/` (read-only): `README.md`; `train/` with inputs, labels and validity masks;
  `public/validation/` with validation inputs; `trusted/validation/` with validation labels for
  aggregate local scoring only; `manifest.json`.
- `/workspace/baseline/temporal-gru/`: the verified baseline, a multimodal GRU with learned
  cumulative hazards, with commands to train it, `package.sh` to build a submission, `publish.py`
  to install one, and RESULTS.md with its measured scores.
- `/workspace/references/`, not verified here:
  - `delphi/`: the upstream Delphi generative-transformer code with synthetic demo data, not
    adapted to this task;
  - `scaffold/`: architecture-neutral data, training and submission scaffolding with explicit
    seams to implement.
- `/workspace/evaluation/`: `score.py` and the scoring library, the same code the verifier uses.
- `/workspace/submission/`: the current package; empty at start.
- `/workspace/research/`: writable space for experiments, checkpoints and notes.

## Rules

1. Train only on the training partition. Use validation only for model selection and aggregate
   error analysis; validation labels must never become training data or features.
2. Do not search for, reconstruct or infer hidden test labels, identities, boundaries or other
   privileged artifacts.
3. Use `sample_id` only for keyed alignment. It, its ordering, row position and other identifiers
   are forbidden predictive features.
4. Respect event and availability cutoffs. Future encounters and observation-boundary metadata are
   forbidden inputs.
5. Preserve the frozen disease order. Emit finite probabilities in `[0,1]` for all 13 horizons,
   nondecreasing over time.
6. The final inference graph is one jointly trained neural model. Internal neural branches and
   experts are allowed; separately trained model ensembles and non-neural prediction components
   are not.
7. Keep claims narrow: this measures recorded inpatient diagnoses under the documented censoring
   and competing-death semantics, not a biological disease mechanism.
8. There is no network access. Vendor every runtime dependency and model asset in the package.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission` from
early on. It contains an executable `infer.sh INPUT_DIR OUTPUT_DIR`, the neural model definition
and the code inference actually runs, the training entry point and frozen configuration, trained
weights, every vocabulary or preprocessing artifact needed offline, and a concise `README.md`.
The README describes the architecture, its structural difference from the baseline, the
validation evidence and the reproduction commands.

The package may be read at any moment, so replace it atomically:

```bash
python /workspace/baseline/temporal-gru/publish.py DIR --check
```

This stages a copy next to the target, runs `infer.sh` on the validation inputs and checks the
output, then swaps it in with one rename. Keep exploratory checkpoints under
`/workspace/research`. The verifier checks packaging and prediction validity; the inference path
and training artifacts are reviewed for architectural compliance.

## Continuous research

There is no end point. The baseline has already been trained and scored in this setup:
`baseline/temporal-gru/RESULTS.md` has its numbers and `reproduction/` its training curve and
per-endpoint validation metrics. Do not train it again to confirm them; check your own data,
training, prediction and scoring path with a short run. Then repeat:

1. Find one architectural limitation or failure mode, per endpoint where it matters.
2. Form a falsifiable hypothesis and change one structural component, representation, objective
   or temporal mechanism.
3. Run the cheapest experiment that can reject or support it, then give promising candidates
   enough training to judge convergence.
4. Record the evidence and publish whenever the validation score improves.

Keep an experiment ledger in `/workspace/research` with hypotheses, commands, runtimes, metrics,
failures and conclusions. Do not stop to hand in a final answer; a published model is a
milestone, not the end of the task.
