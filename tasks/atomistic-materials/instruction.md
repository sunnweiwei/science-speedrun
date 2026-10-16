# Pretrain a crystal potential that stays accurate far from equilibrium

## Task and goal

Develop one machine-learned interatomic potential for inorganic materials: a model that maps atomic
structure to total energy, forces and stress. Train from random initialization; UMA-S is the
starter architecture. No pretrained interatomic-potential weights or teacher predictions may be used.

The evaluator uses your model as an ASE calculator and runs the simulations itself. Every
evaluation below is an objective with equal standing; improve all of them.

| Evaluation | What the evaluator computes with your model | Reference | Metric (lower is better, except F1) |
| --- | --- | --- | --- |
| Stability screening | relax WBM candidates; MP2020-corrected formation energy against the MP hull | Matbench Discovery / WBM PBE | F1 for stable/unstable; hull-distance MAE |
| Phonons | harmonic phonons of relaxed crystals | MDR PBE phonon database | MAE of max/mean frequency, entropy, heat capacity, free energy |
| Elasticity | strain-stress elastic tensors | Materials Project PBE | bulk and shear modulus MAE |
| Thermal conductivity | relax, 2nd- and 3rd-order force constants, Wigner transport at 300 K | PhononDB PBE (Matbench Discovery κ protocol) | κ SRME |
| MOF mechanics | cell relaxation and an equation-of-state scan of metal-organic frameworks, with D3(BJ) dispersion added | MOFSimBench PBE-D3 | bulk-modulus MAPE |
| Amorphous solids | energies and forces of a-C, a-Si, a-SiO₂, a-Li-P-S and a-GeSbTe | AM26 DFT | energy and force RelRMSE |
| Surfaces | cleavage energies of slabs | 36,718-slab PBE benchmark | relative MAE |
| Ion migration | climbing-image NEB for Li hops | LiTraj nebDFT2k PBE | barrier MAE (errors capped at 2 eV) |

Why this is hard: a potential must learn accurate energies, forces and energy derivatives across
near- and off-equilibrium structures. Improvements in stability screening can worsen other physics.
Data selection, mixtures, energy referencing, losses and optimization are part of the task; the
four training sources have different DFT settings and incompatible absolute energy scales.

The headline objective is **Crystal Potential Score v2**, a fixed 0–100 score with equal weight
for all eight suites. All 15 raw metrics remain visible. See the contract for normalization.

## Workspace

- `/workspace/contract.md`: the submission package (`calculator.py` with `make_calculator`), units,
  and how the evaluator runs every suite.
- `/workspace/data/` (read-only):
  - `README.md`: every dataset, its DFT settings, energy convention and exclusions;
  - `pool/`: the training pool, about 42 million labelled DFT frames (91 GB) in ASE-LMDB shards from MPtrj, sAlex,
    MatPES-PBE and OMat24 (rattled and molecular-dynamics frames);
  - `val/<suite>/`: the validation split of every evaluation, with inputs and references;
  - `baselines/mp-standard/`: a convenient v2-filtered MP-compatible starter sample
    (498,841 train / 4,995 labeled-dev frames), drawn from the available pool sources.
- `/workspace/baseline/uma-s-pretrain/`: random UMA-S recipe with training-only elemental offsets
  and normalization. Use it to get a working submission, then improve data selection and mixtures
  from the full pool. Historical adaptation recipes and result tables are comparison material;
  their pretrained checkpoints are not mounted and their scores are not scratch baselines.
- `/workspace/references/`: papers and released models related to each evaluation, for background; their weights may not be used.
- `/workspace/evaluation/`: the verifier's own drivers and scorers: `run_local.py` and `score.py`
  for stability, phonons and elasticity; `extra.py` and `score_extra.py` for the other five.
- `/workspace/submission/`: the current package; empty at start.
- `/workspace/research/`: writable space for derived data, checkpoints and notes.

## Rules

- **One model serves every evaluation.** The same `make_calculator` must work for all eight suites;
  an ensemble counts as one model.
- Train from random initialization; any architecture and size is allowed. Do not load pretrained
  weights, distill pretrained predictions, or import externally fitted training statistics. Fit
  normalizers and energy offsets on your selected training frames only.
- Train only on `/workspace/data/pool` or the provided filtered MP starter sample. Reserve a
  composition-disjoint labeled-dev split for checkpoint selection; do not train on dev or physical
  validation labels. Physical validation is for model selection and diagnostics. Test structures and
  their references are held out from the agent. Evaluation materials are already excluded from the pool by
  composition (AM26 amorphous systems excepted).
- The evaluator controls every relaxation, displacement, strain, NEB and volume scan. Your model
  only returns energy, forces and stress. Do not look up materials, structures or reference values,
  and do not special-case evaluation inputs.
- There is no network access. Everything the calculator needs must be in the package.

## Submission

Keep a complete package that follows `/workspace/contract.md` at `/workspace/submission` from
early on. It must contain:
- `calculator.py` defining `make_calculator(submission_dir, device)`;
- the model weights and any code they need;
- a `README.md` covering the model, training data and mix, compute, validation results on every
  suite, and limitations.

Replace the package atomically: build it elsewhere, check it with the evaluation scripts (a validation
subset is enough), then swap it in.

## Continuous research

There is no end point. The baselines have already been trained and scored in this setup:
`baseline/uma-s-pretrain/RESULTS.md` has every validation metric of the 1M-frame mixed-source scratch
trial (37.08/100), its sample recipe, training curve and costs, and the historical pretrained comparisons
have validation scores and predictions in `baseline/<name>/RESULTS.md` and `baseline/<name>/reproduction/`.
Do not retrain or re-score a baseline, and do not rerun the pretrained models (their weights are not
allowed here). Start from the per-suite gaps in those records and the reference code. Check your own
pipeline with a short run (a few hundred updates, export, a small validation subset) and publish early.

Validation is expensive: scoring one model on all 4,312 validation cases took 2 h 14 min on eight H100s,
and NEB paths take minutes each. Iterate on a fixed subset of every suite (the evaluation drivers accept
any subset of the `data/val/<suite>` inputs, and the comparisons' recorded predictions score on the same
subset). Publish a candidate as soon as it beats the published package on that fixed subset of every
suite; do not hold publication for a full validation run. Run full validation in the background for
mature candidates when the compute is free, and keep its results in the ledger. Then repeat:

1. Find the suite where the model is weakest relative to the references and the baselines.
2. Change one thing: training data selection or mixture, labels and losses, energy referencing
   across DFT settings, schedule, or architecture.
3. Measure on the fixed subset of every validation suite, since a gain on one often costs another,
   and publish the package whenever it improves there.

Keep a concise experiment ledger in `/workspace/research`. Do not stop to hand in a final answer; a
published model is a milestone, not the end of the task.
