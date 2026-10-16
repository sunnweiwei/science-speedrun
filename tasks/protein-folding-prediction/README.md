# Protein Folding Prediction (FoldBench)

## 1. Overview

Continuously research and improve **one 45M--55M model for all nine FoldBench
tracks**, using the supplied FoldingTrain-100K pool and model-independent
ESMC-6B final-hidden INT8 features. No MSA.
The complete development benchmark, native structures, and evaluation code are
visible. Benchmark examples must not be used for training.

The Agent-facing instruction defines a continuing research loop: after every
experiment or candidate publication, it immediately starts the next improvement.

Read [instruction.md](instruction.md) for the experiment rules and
[contract.md](environment/contract.md) for model delivery.

**Status.** This is a task definition with executable preparation, approximately-50M
training, inference, publication, and grading interfaces, not a trained
solution. **Large data are hosted separately on
[Hugging Face](https://huggingface.co/datasets/sunweiwei/ai4sci-foldbench-data)**:
73.1 GB of restored payloads, including the 100K training pool, structures,
complete INT8 features, full evaluation suite and reference-only weights.
[download-source.json](environment/data/download-source.json) pins the dataset
ID, immutable commit and archive-manifest checksum. Only small manifests and
the materialization code belong in GitHub; the actual data live on HF.
No learned weights, training examples, benchmark coordinates, agent credentials,
or experiment logs are included in this repository. The container does pin the
public ESMFold2 CCD dictionary required for all-atom construction.

The starter was trained from scratch and the frozen ESMFold2-Fast reference was replayed by the integrator
(section 6). The workspace follows [TASK_STANDARD.md](../TASK_STANDARD.md).

## 2. Data

The Dockerfiles build the development and independent verification environments
from public sources. They install the pinned FoldBench scorer, Biohub ESMFold2
runtime and CCD, plus pinned Protenix, OpenDDE, and DPLM source trees. They do not
download training data or learned model weights.

Use Python 3.11+ and a current HF CLI supporting `hf download` (publication
tested with 1.32.0). No credentials are required to download this public release.
The committed defaults are ready to use; do not replace them with `main`.

```bash
export TASK_DATA_DIR=/absolute/path/on/large-disk/foldbench-data
bash protein-folding-prediction/environment/data/materialize.sh "$TASK_DATA_DIR"
```

The materializer downloads 75 tar shards, checks each archive and each restored
file against SHA-256 inventories, then audits the frozen scientific release.
It restores the existing five-directory layout without changing any payload.
Allow at least 76 GB of free space (decimal; extra HF cache space may be needed).
Each downloaded shard is removed after successful extraction; `--keep-archives`
retains an additional 73.3 GB. Rerunning checks existing files and downloads
only shards with missing payloads. Corrupt existing files are reported, never
silently overwritten. `--verify-only` rechecks a completed release offline.
An explicit mirror override may use `FOLDBENCH_HF_REPO` and
`FOLDBENCH_HF_REVISION`, but it must contain the exact pinned payloads.

During local release preparation,
`environment/data/finalize_local_features.py DATA_ROOT` requires complete
training and suite fixed-feature coverage, updates the two feature indexes, and
stages the suite feature subtree. Then
`assemble_local_release.py ORGANIZER_ROOT RELEASE_ROOT` creates a clean,
hardlinked release view containing only Agent-visible data; point
`TASK_DATA_DIR` at that release root. `validate_release.py` performs the
fail-closed release audit. None of these helpers generates features or accesses
model weights.

The native Compose overlays require this existing directory. Development mounts
five allowlisted read-only subtrees under `/workspace/data`: training manifests, structures,
fixed features, FoldBench, and reference models (the published manifests store absolute `/data/...`
paths, so the image links `/data` to `/workspace/data`). It deliberately does not mount
the organizer's HF cache or ESMC-6B checkpoint. The separate verifier mounts
**only `foldbench/`**, including fixed suite input features, at `/benchmark`. It
does not receive training data, ESMFold2-Fast, or the live workspace. Harbor
transfers the declared `/workspace/submission` artifact.
Other environment providers may supply equivalent mounts instead of Compose.

An ordinary Harbor invocation can then use `-p protein-folding-prediction` and the operator's
chosen agent/model. This repository does not choose or install an agent, manage
authentication, or implement a runner. `task.toml` sets the standard 48 h external agent budget (the controller may override it);
the default verifier timeout is configurable separately.

`instruction.md`, `contract.md`, the data README, `baseline/`, `references/` and development
evaluation are explicitly copied into `/workspace` by the environment build (build context: the
task directory). Merely adding a file at the task root does not make it available inside a
container.

## 3. Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md`, `/workspace/contract.md` | the prompt; the replay contract (package, inputs, outputs, reward) |
| `/workspace/data/` | `README.md`; `manifests/`, `objects/`, `features/`, `foldbench/`, `models/` (read-only mounts; `/data` links here) |
| `/workspace/baseline/starter/` | the approximately-50M ESMFold2 graph from scratch: audit, index, 8-GPU training, continuous atomic publication, replay; reproduced reward **0.037** (ESMFold2-Fast reference **0.545**) |
| `/workspace/references/` | runnable approximately-50M architectures, pinned upstream sources, prior experiments (not re-run by the integrator) |
| `/workspace/evaluation/` | direct development evaluation with the native FoldBench scripts |
| `/workspace/submission/`, `/workspace/research/` | empty; the current package, and writable scratch on a large disk |

## 4. Evaluation

[Development evaluation](environment/evaluation/README.md) is self-service. The
agent may run the native FoldBench scripts directly, choose subsets, and inspect
errors; it need not call a service or a prescribed development wrapper.

[The verifier](tests/README.md) snapshots and replays the currently published model in a fresh environment
and reports one numeric score per track to Harbor, plus `reward`, their unweighted mean
(TASK_STANDARD.md). Missing/invalid predictions remain in the original target denominators.
The full canonical suite contains 1,522 assemblies and 1,823 target rows, with
seed 101 and one sample. `environment/data/split_suite.py` (run by `materialize.py`) splits it
into `foldbench-val` (74 assemblies, 108 rows: up to ten examples per track) and `foldbench-test`
(1,448 assemblies, 1,715 rows). Related assemblies stay on one side: the same PDB entry, or an
identical or similar chain (k-mer Jaccard >= 0.3), with a fixed seed. The agent mounts only
`foldbench-val`, with its native structures, for development; the verifier mounts only
`foldbench-test`. An inference package must nevertheless handle arbitrary
new requests of the same schema; it may not depend on these IDs or counts.

## 5. Running

```bash
export TASK_DATA_DIR=/large-disk/foldbench-data TASK_PRIVATE_DIR=/large-disk/foldbench-data   # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/foldbench-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p protein-folding-prediction -a <agent> -m <model> --allow-agent-host <model API host>
```

The development environment asks for eight H100s. The verifier also asks for eight, with a
24-hour timeout; replaying the full suite took 1 h 45 min of inference on one H100. The contract
bounds the package's inference at 8 h and 256 GB of host memory (`task.toml` sets the verifier's
`memory_mb`); `tests/grade.py` kills an `infer.sh` that runs longer, with its whole process group,
and scores the package 0 as an invalid submission. An agent package whose CPU fallback for long
assemblies needed more than 200 GB and about 150 h was the reason. The unit tests run in the
verifier image.

## 6. Results and validation

Independent end-to-end run on one 8× H100 80 GB host from the committed
repository (images built unmodified from `main`; the 73.1 GB release
re-verified file by file; unit tests 43/43). Full details and artifacts are in
the integrator's review notes; the numbers below are the ones that matter.

**Evaluator, full 1,522-request suite (measured before the validation/test split), three known answers**

| Submission | monomer_protein lDDT | protein–protein DockQ ≥ 0.23 | antibody–antigen | other tracks |
| --- | ---: | ---: | ---: | --- |
| Native structures replayed as predictions (plumbing oracle) | 0.9996 | 99.3 % | 96.5 % | RNA/DNA 1.0; peptide 94.1 %; protein–RNA 100 %; protein–DNA 96.4 %; ligand 99.8 % |
| Frozen official ESMFold2-Fast (organizer side, needs ESMC-6B; 1,470/1,522 folded, 52 exceed 80 GB) | 0.822 (0.829 over finite targets; published 0.8375) | 55.2 % (62.9 % over finite; model card 68) | 29.1 % (31.7 % over finite; model card 50, multi-sample) | RNA 0.569; DNA 0.372; peptide 60.8 %; protein–RNA 52.9 %; protein–DNA 64.8 %; ligand 51.1 % |

The oracle row is the ceiling under the strict denominators: the pinned
official scorer returns no finite DockQ for about 24 native interfaces (12
protein–DNA rows raise inside DockQ-v2), so no submission can exceed those
values. Scoring the full suite takes ~10 minutes; the ESMFold2-Fast row was
produced with one sample, seed 101, three loops.

Interface success also requires CAPRI acceptable geometry (ligand RMSD ≤ 10 Å or interface RMSD
≤ 4 Å) since 2026-10-07. An agent package collapsed every chain onto the others: all native
contacts were recovered (DockQ ≈ 0.36 from fnat alone) at interface RMSD ≈ 15 Å and ligand RMSD ≈ 30 Å,
and the interface tracks scored 0.86–1.0 (reward 0.524) with every monomer at 0. Under the rule the
same package scores 0.011. On the raw scores of an ESMFold2-Fast replay the interface rates fall by
0.01–0.04 (reward 0.569 → 0.557); the native-structure oracle is unchanged (0.985). The rows above
predate the rule; the starter's interface rates have not been rescored yet.

**Starter baseline (`train.sh` defaults, one epoch = 16,334 steps, 6× H100,
3.7 h wall including one resume; loss 0.80 → 0.24; package at step 16,334)**

| Track | Score | Finite / targets | Diagnostic (from the scorer's raw rows) |
| --- | ---: | ---: | --- |
| monomer_protein | 0.000 | 332 / 334 | TM-score 0.530 (max 0.90), RMSD 9.2 Å |
| monomer_rna | 0.000 | 15 / 15 | TM-score 0.10 |
| monomer_dna | 0.000 | 14 / 14 | TM-score 0.08 |
| interface_protein_protein | 0.057 | 254 / 279 | mean DockQ 0.078 |
| interface_antibody_antigen | 0.017 | 162 / 172 | mean DockQ 0.040 |
| interface_protein_peptide | 0.118 | 44 / 51 | mean DockQ 0.126 |
| interface_protein_rna | 0.086 | 70 / 70 | mean DockQ 0.086 |
| interface_protein_dna | 0.058 | 319 / 330 | mean DockQ 0.106 |
| interface_protein_ligand | 0.000 | 486 / 558 | lDDT-PLI 0.24, RMSD 9.2 Å |

Replay of this package through `tests/test.sh` on one H100 took 1 h 45 min of
inference (1,498 structures; 24 assemblies above the replay limit recorded as
failures) plus ~10 minutes of scoring.

Read the monomer rows carefully: **lDDT is exactly zero although the global
folds are often right** (7pv5: TM-score 0.756, RMSD 2.0 Å, lDDT 0.0). The
pinned OpenStructure scorer applies stereochemistry checks, and the one-epoch
starter's local geometry is physically invalid (N–CA bonds of 0.8 Å, C–O of
0.4 Å, clashes in most residues), so every residue's lDDT contribution is
zeroed while TM-score and RMSD, which only superpose Cα atoms, look fine. The
same pattern appears in every earlier one-pass 50M reference run (lDDT 0.00,
TM ≈ 0.6). Until a candidate's bond geometry is valid the three monomer tracks
read zero whatever its fold quality; geometric validity (bond/angle terms,
post-hoc relaxation) is therefore the first lever, and the raw TM-score/RMSD
columns are the diagnostic to watch meanwhile.

Fixes that came out of this acceptance (this branch): per-request failure
isolation and a low-memory replay path in the starter's `predict.py`/runtime
(the original starter aborted the whole verifier run on the first CUDA
out-of-memory assembly), a guarded training step and a pool filter for the 71
polymer-less entries (one of which crashed the run at step ~8,600), and an
executable `tests/test.sh`.

Earlier pipeline validation and the data publication record are in
[`docs/VALIDATION.md`](docs/VALIDATION.md).

## 7. Limitations

- The development suite is fully visible (inputs, natives, scorer). Scores are
  **benchmark-visible development scores**, not blind holdout results.
- The visible verifier runs predictor and scorer sequentially in one container. It is not a hardened
  boundary against code that reads native structures; a private evaluator must isolate them
  (`environment/contract.md`).
- About 24 native interfaces have no finite DockQ under the pinned scorer, so even the oracle scores below 1
  on those tracks.

## 8. Citation and licenses

- The primary runnable starter is a 49,198,221-parameter, from-scratch Biohub
  ESMFold2 graph adapted to the current `[L,2560]` final-hidden input. It includes
  8-GPU BF16 training, strict resume, continuous publication, and offline replay.
- Reference provenance: the compact smoke helpers came from
  `sunnweiwei/protein-folding` at `09900626ae9335df610cddbc0801a2acdc0a7bd4`;
  the restored validated architecture workspace came from clean paths at
  `1e21e527914a49b02acb4b79bcbdada805944cd5`. Needed source is vendored, so no
  private-repository credential is required.
- Official scorer: [FoldBench at 4273f687](https://github.com/BEAM-Labs/FoldBench/tree/4273f6877d82bd0b2fa476d1b2f34d121cbccc70).
- [Harbor task format](https://www.harborframework.com/docs/tasks).
- [HF download interface](https://huggingface.co/docs/huggingface_hub/guides/download).

Reference repositories and weights retain their upstream licenses. The current
instruction and contract deliberately restore the validated 45M--55M scientific
target while retaining the Harbor task's end-to-end and continuously published
model-development workflow.

