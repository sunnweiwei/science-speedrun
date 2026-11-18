# Develop one from-scratch crystal generator

## Task and goal

Develop one generative model of inorganic crystal structures, trained from random initialization
on the MP-20 training split, that generates stable, unique and novel structures without target
conditions. Generators that propose new stable materials shorten the search for new materials.

The package receives unconditional requests (`/workspace/contract.md`). The evaluation asks for
2,500 fresh structures with an unseen seed. The grader first relaxes a separate copy of each submitted crystal with the frozen UMA
`uma-s-1` (`omat`), BFGSLineSearch and FrechetCellFilter, up to 200 steps at 0.02 eV/Å and
300 seconds per crystal. It retains finite unconverged iterates and uses the original
candidate when refinement fails. It then scores the resulting geometry with ORB-v3, MACE-MP
and UMA. Mean energy above hull defines stable (≤0 eV/atom) and metastable (0–0.1 eV/atom)
bins, with stable structures excluded from the metastable bin. Let `a = SUN / 2,500` be the strict
stable-unique-novel yield (`ΔE_hull ≤ 0 eV/atom`) and `b = (SUN + MetaSUN) / 2,500` be the stable-or-metastable
unique-novel yield (`ΔE_hull ≤ 0.1 eV/atom`), each divided by all 2,500 requested slots.

For equal compositions, CrystalGen applies MatterGen's v1.0.3 ordered-disordered matcher.
The additional dilute-parent guard identifies a known lattice decorated with a substituent
occupying ≤25% of the host-plus-guest sites. It requires relative radius difference ≤0.3 and
electronegativity difference ≤1.0. Supercells, symmetry copies and dilute variants of a known
compound such as GdDy7Cu8 from DyCu are covered by these comparisons. A cross-element substitution requires both the measured radii and the electronegativities. Ordinary same-species geometric matches still apply to materials such as XeF₂ when a radius is unavailable.

The task primary objective (`score` and `reward` in `reward.json`, in `[0, 1]`) is the equal mean of strict SUN yield and stable-or-metastable SUN yield:

```text
score = reward = (a + b) / 2 = (SUN + 0.5 * MetaSUN) / 2,500
```

Validity, uniqueness, novelty, strict SUN (`unconditional_sun`), MetaSUN (`unconditional_metasun`), and unweighted `SUN + MetaSUN` (`unconditional_sun_metasun`) are reported separately, along with formation-energy, relaxation-RMSD, and element-arity (`n_elements`) diagnostics in `report.json` (diagnostics only, never used as gates). Missing or invalid outputs stay in the denominator and count as failures.

The trained 2,000-epoch OMatG starter scores **0.0892** (`(8 + 0.5 × 430) / 2,500`; unweighted `SUN + MetaSUN = 0.1752`) on its 2,500 archived requests (8 SUN and 430 MetaSUN). The aggregate-statistics smoke generator is the no-learning control. The starter directory includes its input, generated and refined CIFs, and score receipts.

The optional `--as-submitted` evaluation scores the raw lattice and coordinates. It also
measures displacement from the raw geometry under all three potentials on 250 prespecified
slots (50 steps, 0.02 eV/Å), with coverage and a 95% structure-bootstrap interval.
This result is recorded separately from the primary UMA-200 reward.

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
  sampling. Return the candidate coordinates directly. The scoring potentials cannot be part of the package.
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
and scored under the historical UMA-200 protocol: `baseline/starter/RESULTS.md` has its numbers and `reproduction/` its
2,500 generated and relaxed structures with the full score report. Do not retrain it to confirm
them. Take your own pipeline through a one-step fit, generation of real CIFs, publication,
relocation and development scoring, and publish a trained candidate.

## Continuous research

There is no end point. Analyze where the current generator fails: validity, uniqueness, novelty,
stability after the fixed relaxation, and the composition, space-group and size distributions. Form a
hypothesis, train and score a candidate on development batches of 100 or 2,500 structures, and
publish it when it is better. Then start the next experiment. Do not stop to hand in a final
answer. A published model or a reached score is a milestone, not the end of the task.
