# Crystal Potential Pretraining

## 1. Overview

Train one conservative interatomic potential **from random initialization**, using UMA-S as the
starter architecture. Atomic numbers, positions and periodic cell go in; total energy, forces and
stress come out. The evaluator uses this potential in eight fixed physics calculations.

**Scientific purpose:** pretrain a materials potential that supports accurate screening and
physical simulation across different materials and atomic configurations. The task measures
whether choices about data, energy referencing and training improve the model's use in those
calculations. UMA-S supplies the architecture; its backbone and prediction head start with random
weights, and the recipe loads no pretrained UMA checkpoint.

| | Crystal generation | Crystal potential pretraining |
| --- | --- | --- |
| Goal | Propose new crystal structures | Learn energies and derivatives that make simulations accurate |
| Input → output | Unconditional sampling request → candidate crystals | Atomic structure → energy, forces, stress |
| Current benchmark starter | OMatG trained from scratch | Random UMA-S starter; other architectures allowed |
| Success | Stable or metastable, unique, novel structures | Accurate screening, vibrations, mechanics, transport and barriers |

This tests a component needed for materials discovery; a high score does not establish a newly
synthesizable or useful material. The expanded data/evaluation and Harbor layout come from PR #90;
this track requires scratch training and adds a fixed eight-suite score.

## 2. Data

| Role / source | Frames | Use |
| --- | ---: | --- |
| Training pool: MPtrj | 1,065,435 | MP-compatible PBE/PBE+U |
| Training pool: sAlex | 9,434,535 | MP-compatible PBE/PBE+U |
| Training pool: MatPES-PBE | 193,171 | PBE, no +U |
| Training pool: OMat24 | 31,716,685 | PBE/PBE+U, newer pseudopotentials; 35% of upstream shards |
| **Full training pool** | **42,409,826 (~91 GB)** | Select/mix within the compute budget |
| Starter training subset | 498,841 (1.18% of pool) | Filtered MP-compatible sample |
| Starter labeled dev | 4,995 | Checkpoint selection; excluded from fitting statistics |

- [Pinned public release](https://huggingface.co/datasets/sunweiwei/ai4sci-atomistic-materials-data/tree/2cffee553a107c9fe8d9799b2e6b79cc5002f914), with checksummed artifacts; no data or weights in Git.
- Pool excludes evaluation compositions/parents, except common AM26 amorphous compositions.
- Training sources have different energy scales; referencing and mixture selection are part of the task.
- Random weights only. Fit normalizers and elemental offsets on training data only; do not train on
  labeled dev or physical-validation labels. Historical pretrained weights are not mounted.
- [Data details](environment/data/README.md), [materialization](environment/data/MATERIALIZE.md),
  [scratch starter](environment/baseline/uma-s-pretrain/README.md).

## 3. Workspace

The agent sees the pool, filtered MP starter sample, validation data, scratch starter code and
fixed evaluation drivers. `/workspace/submission` holds its ASE calculator package;
`/workspace/research` is writable. Historical recipes/results remain for reference, without
pretrained weights. [Instructions](instruction.md) and [contract](environment/contract.md).

## 4. Evaluation

| Suite | Primary metric(s) | Validation | Test |
| --- | --- | ---: | ---: |
| Matbench Discovery / WBM | Stability F1 ↑ | 1,000 | 5,000 |
| MDR phonons | Mean-frequency MAE ↓ | 300 | 500 |
| MP elasticity | Bulk / shear modulus MAE ↓ | 300 | 500 |
| PhononDB thermal conductivity | κ SRME ↓ | 40 | 63 |
| MOFSimBench mechanics | Bulk-modulus MAPE ↓ | 40 | 60 |
| AM26 amorphous solids | Energy / force RelRMSE ↓ | 376 | 558 |
| Cleavage energies | Relative MAE ↓ | 2,000 | 2,000 |
| Li-migration NEB | Barrier MAE ↓ | 256 | 390 |
| **Total** | **8 suites; 15 raw metrics** | **4,312** | **9,071** |

- **Crystal Potential Score v2:** 0–100, higher is better; each suite has equal weight regardless of
  case count. Divide primary errors by fixed validation reference scales, average within suite,
  then geometrically average `100 / (1 + error)` across eight suites.
- 50 matches the rounded historical MP 3-epoch validation reference profile; 100 is ideal.
  Missing/invalid primary metrics give 0. Per-case failures retain the inherited penalties.
- All 15 raw metrics remain available, including hull MAE and additional phonon diagnostics.
- Custom task subsets/score, not official leaderboard submissions. Constants and commands are in
  the [contract](environment/contract.md); `reward.json` includes the score and `reward = score / 100`.
- Offline verifier: untrusted calculator sees only test inputs; a separate grader scores sealed
  predictions against references. Train/val/test roles are fixed; public references cannot provide
  a cryptographic secrecy guarantee.

## 5. Running

```bash
bash environment/data/materialize.sh release /data/atomistic-materials
export TASK_DATA_DIR=/data/atomistic-materials TASK_PRIVATE_DIR=/data/atomistic-materials
export TASK_RESEARCH_DIR=/scratch/am-run-001
harbor run --task atomistic-materials ...
```

Only pool, filtered starter sample and validation are mounted for the agent. The release also
contains historical models for operators; these are deliberately not mounted. Build/launch details
are in [MATERIALIZE.md](environment/data/MATERIALIZE.md).

## 6. Results and validation

| Item | Budget or measured evidence |
| --- | --- |
| Agent budget | 48 hours, 8×H100, 128 CPUs, 768 GB RAM |
| Test inference budget | 3 hours across all eight suites (unfinished cases score as failures); verifier timeout 4 hours |
| Full eight-suite test evaluation | **2 h 5 min** on 8×H100 with all suites running at once (mixed-replay model, 2026-10-15); 6 h 18 min when the suites ran one after another (PR #90) |
| Historical adaptation training | 2 h 16 min MP fine-tune; 2 h 17 min mixed replay |
| Completed mixed-source scratch trial | 1,000,000 train / 5,000 labeled-dev frames; 10 epochs; validation Score v2 **37.08/100** |
| Scratch training / validation GPU allocation | **≈17 h 55 min / 2 h 14 min**, each on eight H100s |

Historical pretrained results and verifier measurements are retained in
[VALIDATION.md](docs/VALIDATION.md) and `environment/baseline/*/RESULTS.md`.
They are comparisons, not allowed initializations or scratch scores.

The **30-epoch scratch pilot was stopped by the user** at ~19.60 epochs, with checkpoints retained.
It used the frozen 500k/5k sample and planned 1,200-case three-suite validation with **Score v1**.
That pilot's sample predates the added exclusions
(1,159 train / 5 dev frames are removed in the new starter sample), so do not report it as a clean
8-suite result. Its 6,000-case test remains unused by that trial; PR #90 has already published
historical baseline test results. [Scratch validation record](docs/VALIDATION-pretraining.md).

The separate **10-epoch mixed-source trial is complete** on all 4,312 validation cases. Its
9,071-case test remains unused. The final curve, all raw metrics and timing scopes are in the
scratch validation record. The packaged MP-compatible starter uses a different sample; the
mixed-source result is not a measured score for that starter recipe.

## 7. Limitations

The small κ/MOF panels limit precision. AM26 has no upstream license file at the pinned commit.
The scratch trial passed cloud validation; replay through the full Docker/GPU verifier remains
pending. The mixed-source preparation and execution recipe remains in operator storage; the
packaged starter does not reproduce that trial as written. Prior baselines include v1 overlap
and a historical mixed-replay split bug; see their result notes.

## 8. Citation and licenses

[Licenses and citations](docs/THIRD_PARTY_NOTICES.md). UMA/OMat24, MPtrj/CHGNet, MatPES,
Matbench Discovery/WBM, MDR, PhononDB, MOFSimBench, AM26, the cleavage benchmark and LiTraj.
