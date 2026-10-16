# Protein Sequence Modeling (PEER)

## 1. Overview

Develop **one unified protein sequence model** for eleven tasks drawn from the
[PEER benchmark](https://arxiv.org/abs/2206.02096) (Xu et al., NeurIPS 2022
Datasets and Benchmarks): protein function regression, localization, fold and
secondary-structure classification, and protein–protein interaction prediction.
The agent trains a shared backbone with task heads, packages a frozen
submission, and an independent verifier replays it on the PEER test splits.

Included PEER tasks (all sequence-only):

| Task | Type | Eval metric | Eval split |
| --- | --- | --- | --- |
| `fluorescence` | regression | Spearman ρ | test |
| `stability` | regression | Spearman ρ | test |
| `beta_lactamase` | regression | Spearman ρ | test |
| `solubility` | 2-class | accuracy | test |
| `subcellular_localization` | 10-class | accuracy | test |
| `binary_localization` | 2-class | accuracy | test |
| `fold` | 1195-class | accuracy | test_fold_holdout |
| `secondary_structure` | per-residue 3-class | accuracy | cb513 |
| `yeast_ppi` | pair 2-class | accuracy | test |
| `human_ppi` | pair 2-class | accuracy | test |
| `ppi_affinity` | pair regression | RMSE | test |

Excluded from PEER: contact prediction (requires structure coordinates and a
much larger data footprint) and the two protein–ligand tasks PDBbind and
BindingDB (require a second, small-molecule input modality). The extra holdout
splits (fold family/superfamily holdouts, SSP casp12/ts115, PPI
cross-species) are not materialized.

The primary reward is the equal-weight mean of eleven per-task rewards
(accuracy for classification, `max(ρ, 0)` for Spearman tasks,
`exp(-RMSE / fit_std)` for `ppi_affinity`). See
[environment/contract.md](environment/contract.md).

**Status.** Data verified end to end on real archives. The ESM-2 starter recipe was reproduced by
the integrator on H100s: 0.6695 and 0.6774 (seeds 0 and 1), against 0.6368 documented. The
workspace follows [TASK_STANDARD.md](../TASK_STANDARD.md).

## 2. Data

All PEER source archives are the pinned public PEER/TAPE distributions on S3
(URL + md5 recorded in
[environment/data/materialize.py](environment/data/materialize.py)); data is
not redistributed in Git. The operator runs
`environment/data/materialize.sh TARGET_DIR` once to download, verify, and
convert everything to the `peer-protein-input/v1` JSONL schema, then
`build_pretrain_subset.py` to build the unlabeled pretraining corpus. See
[environment/data/README.md](environment/data/README.md) (what the agent sees) and
[environment/data/MATERIALIZE.md](environment/data/MATERIALIZE.md) (how the controller prepares it).

**Training pool scale.**

| Pool | Examples | Sequences | Residues | Purpose |
| --- | --- | --- | --- | --- |
| `pretrain/uniref50_subset.fasta` | — | 37,294,039 | 9,893,755,312 | self-supervised pretraining |
| `fit/<task>.jsonl` | 219,325 | 262,360 | ~76M | supervised training |
| `tune/<task>.jsonl` | 23,395 | 24,008 | ~8M | model selection |
| held-out eval (verifier only) | 49,297 | 50,254 | ~14M | scoring |

All counts are measured from the pinned archives, not taken from torchdrug's
docstrings — those disagree with the distributed data for 16 of 33 splits (see
[the data README](environment/data/README.md#split-counts-measured)).

For comparison, [protein-folding-prediction](../protein-folding-prediction/) trains a ~50M-parameter model on a
100,000-record pool of structures. This task's supervised pool is about twice
that in examples, and the pretraining corpus is all of UniRef50 in the 30–1024
residue range — a sequence model needs far more examples per parameter than a
structure module, and unlabeled protein sequence is the one resource here that
is free to scale.

The corpus is not subsampled, because a candidate never needs to finish an
epoch: pretraining draws minibatches, and seeing part of a large corpus once
beats seeing all of a small one repeatedly. Measured on one A100 with
length-bucketed token-budget batches, an `esm2_t6_8M`-scale encoder sustains
**495K tokens/s** and covers the whole corpus in 5.6 h, a 35M-scale one in
14.0 h — so a research loop can iterate at small scale within hours, and the
choice between model size and training length is an experiment rather than a
constraint.

One practical warning worth reading before spending the budget: over this
length distribution, fixed sequence-count batches waste **3.4× of the compute**
on padding. Batching by token budget over length-sorted buckets — which is what
ESM-2 does — measured 4× faster end to end.

The pretraining subset is **decontaminated against the held-out eval sequences
at 50% identity** using the same MMseqs2 procedure ESM-2 applied to its own
validation split. This matters: PEER's test proteins are natural sequences that
appear in UniRef50, so an unfiltered corpus would leak the test set into
pretraining.

## 3. Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md` | the prompt |
| `/workspace/contract.md` | `peer-protein-input/v1`, package location and invocation, prediction formats, scoring |
| `/workspace/data/` | `README.md`; `fit/`, `tune/`, `pretrain/`, `manifest.json`, `stats.json` (read-only mounts) |
| `/workspace/baseline/starter/` | ESM-2 pretraining + joint fine-tuning + `tune` selection + atomic publish; reproduced **0.6695 / 0.6774** |
| `/workspace/baseline/trivial/` | per-task constants; **0.2874** |
| `/workspace/references/` | the author's measured runs (37M generic transformer, throughput per size) and model specs |
| `/workspace/evaluation/` | self-scoring on `tune` with the grader's metric |
| `/workspace/submission/`, `/workspace/research/` | empty; the current package, and writable scratch on a large disk |

## 4. Evaluation

The verifier has two services:

- **inference:** replays `infer.sh` on the held-out inputs (`public/verifier`, no labels) as an
  unprivileged uid, with no network;
- **grader:** a separate service that reads the labels (`private/`) and writes
  `/logs/verifier/reward.json`. It contains `reward`, the equal-weight mean of the eleven task
  rewards: accuracy for classification, `max(ρ, 0)` for Spearman tasks, and
  `exp(-RMSE / fit_std)` for `ppi_affinity`.

`tests/README.md` states the isolation boundary and its limits.

## 5. Running

```bash
export TASK_DATA_DIR=/path/to/peer-data TASK_PRIVATE_DIR=/path/to/peer-data     # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/peer-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p protein-sequence-modeling -a <agent> -m <model> --allow-agent-host <model API host>
```

- **External budget:** set by the controller; `task.toml` sets the standard 8× H100, 48 h agent budget (64 CPUs, 512 GiB). The baseline recipe uses one GPU.
- **Reference solution:** `solution/solve.sh` publishes the trivial control (Harbor's `oracle`
  agent).
- **Unit tests:** `bash tests/run_unit_tests.sh`.

## 6. Results and validation

| system | reward | who measured it |
| --- | ---: | --- |
| trivial control | 0.2874 | task author |
| 37M generic transformer, 5M-sequence corpus | 0.6147 | task author (`environment/references/measured_runs.md`) |
| ESM-2 8M starter, 5M-sequence corpus, A100 | 0.6368 | task author |
| ESM-2 8M starter, the same recipe on one H100 | 0.6695 / 0.6774 (seeds 0 / 1) | integrator (`environment/baseline/starter/RESULTS.md`) |

The smoke run, the reference-baseline records and the container verification are in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## 7. Limitations

**Comparability with the PEER paper.** Do not compare scores from this task against the numbers in the PEER paper.
They are not measured on the same data.

The paper's per-task results were computed on splits whose sizes match
torchdrug's dataset docstrings, and those docstrings disagree with the archives
torchdrug actually distributes for 16 of the 33 splits — most starkly for the
three PPI tasks, where the documented figure counts unique proteins while the
data contains five times as many labeled pairs. This task uses the distributed
data, with counts measured directly (see
[the data README](environment/data/README.md#split-counts-measured)).

Three further differences compound this:

- **The metric is not PEER's.** Rewards here are bounded to [0,1] and
  aggregated as an equal-weight mean across eleven tasks so a single number can
  drive the grader. PEER reports each task in its native units.
- **The eval split is one per task**, chosen as the PEER-standard one; the
  additional holdouts (fold family/superfamily, SSP casp12/ts115, PPI
  cross-species) are not materialized.
- **The training rules are stricter.** Submissions must train from random
  initialization on the provided corpus, with no external pretrained protein
  models. Much of the PEER leaderboard uses pretrained encoders.

The reference to beat is therefore a baseline trained under this task's own
rules and budget, not a literature number. Reporting a score here as if it were
comparable to a published PEER result would be wrong in both directions.

## 8. Citation and licenses

If you use this task, cite PEER (Xu et al., 2022) and the upstream dataset
papers listed in [environment/data/README.md](environment/data/README.md),
including TAPE (Rao et al., 2019) for the fluorescence, stability, secondary
structure, and remote homology (fold) data.
