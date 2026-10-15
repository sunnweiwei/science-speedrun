# Develop one from-scratch crystal generator

## Task and goal

Develop one generative model of inorganic crystal structures, trained from random initialization
on the MP-20 training split, that generates stable, unique and novel structures without target
conditions. Generators that propose new stable materials shorten the search for new materials.

The package receives unconditional requests (`/workspace/contract.md`). The evaluation asks for
2,500 fresh structures with an unseen seed. It relaxes each one with frozen UMA `uma-s-1`
(BFGSLineSearch + FrechetCellFilter, `fmax = 0.02` eV/Å, at most 200 steps), then scores the
relaxed set with the pinned LeMat-GenBench ensemble (ORB, MACE and UMA hulls, LeMat-Bulk novelty).
Novelty and uniqueness account for substitutional disorder, as in MatterGen: a structure is not
novel if merging a dilute substituent into a chemically similar host (relative radius difference
≤ 0.3, electronegativity difference ≤ 1.0, substituent ≤ 1/4 of the pair's sites) gives a known
LeMat-Bulk structure, and two such variants of the same parent count as one. Ordered doping of
known compounds is not a new material.
The reward is the SUN + MetaSUN fraction:

```text
reward = (stable-unique-novel count + metastable-unique-novel count) / 2,500
```

Validity, uniqueness, novelty, SUN and MetaSUN are reported separately. Missing or invalid outputs
stay in the denominator and count as failures.

| generator | SUN + MetaSUN |
| --- | ---: |
| smoke generator (aggregate statistics, no learning) | 0.106 |
| OMatG starter, 2,000 epochs from scratch (baseline) | 0.208 |

Both rows were scored before the disorder-aware novelty; it changes about 1 % of the starter's
structures. About half of the starter's score is reached without learning, so judge improvements
against both rows. Aim well above the starter.

## Workspace

- `/workspace/contract.md`: the request format, the output manifest, the model manifest, and the
  package and inference limits.
- `/workspace/data/`:
  - `README.md` describes the data;
  - `mp20/train.csv` (27,136 structures), `mp20/val.csv` (9,047) and `mp20/manifest.json`;
  - `scorer-assets/`: LeMat-Bulk, the MLIP hull tables and the ORB, MACE and UMA weights, used
    only by the evaluation.

  All of these are read-only.
- `/workspace/baseline/starter/`: the verified unconditional OMatG starter (`omg_port/`: fit,
  publish, generate) and the smoke generator, with README and reproduced results.
- `/workspace/references/`: the OMatG and LeMat-GenBench sources in the image, the task author's
  historical results and papers, not re-run here.
- `/workspace/evaluation/`: the exact development evaluation (`score.sh`, `relax.sh`) and a
  100-structure example request. The shared scoring code is in `/opt/crystalgen/scoring`, the
  OMatG source in `/opt/omg`, the scorer in `/opt/lemat`.
- `/workspace/research/`: writable space for runs, caches, generated batches and notes.

## Rules

- Fit parameters only on `mp20/train.csv`. Use validation for model selection, never for fitting.
- Initialize every submitted parameter from scratch in this task. Do not use published generator
  checkpoints, external structure databases, generated pseudo-label corpora, or outputs of the
  scoring potentials as training or distillation data.
- The scorer assets are evaluation tools. Do not copy them, index them, or make the package depend
  on them.
- Use one parameter set for all requested samples, with at most 70,000,000 learned parameters at
  inference.
- The generator must be learned. Do not replace it with a structure lookup table or a hand-written
  database enumerator.
- You may change the architecture, objectives, optimization, data sampling and inference-time
  sampling. The relaxation belongs to the grader: the package does not ship a relaxer and cannot
  change its settings.
- During `generate.sh`, do not train, change weights, contact a service or leave processes
  behind. Write every requested structure; a missing or invalid structure is a failure, not a
  smaller denominator.
- There is no network.

## Submission

Keep a complete package at `/workspace/submission` from early on: `generate.sh`,
`model_manifest.json`, a concise `README.md`, model code, configuration and the one selected
from-scratch checkpoint (`.safetensors`). Keep optimizer state, old candidates and scorer assets
out of it, under `/workspace/research`.

The package may be read at any moment, so replace it atomically, as the starter's `publish.py`
does. First run the smoke path to check the contract. The OMatG starter has already been trained
and scored in this setup: `baseline/starter/RESULTS.md` has its numbers and `reproduction/` its
2,500 generated and relaxed structures with the full score report. Do not retrain it to confirm
them. Take your own pipeline through a one-step fit, generation of real CIFs, publication,
relocation and development scoring, and publish a trained candidate.

## Continuous research

There is no end point. Analyze where the current generator fails: validity, uniqueness, novelty,
stability after relaxation, and the composition, space-group and size distributions. Form a
hypothesis, train and score a candidate on development batches of 100 or 2,500 structures, and
publish it when it is better. Then start the next experiment. Do not stop to hand in a final
answer. A published model or a reached score is a milestone, not the end of the task.
