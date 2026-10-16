# PDE surrogate: cross-equation generalization on regular grids

![What the task is: four unrelated families of 2D PDEs, four context frames in and a free-running rollout out, and why the transfer question is still open](docs/science_intro.png)

*Regenerate with `python3 docs/make_science_intro.py --work <verifier work dir>`. A frame-by-frame view of all four sources is in [`docs/task_overview.png`](docs/task_overview.png) (`docs/make_task_overview.py`).*

## 1. Overview

**Task.** Build one autoregressive surrogate for 2D time-dependent PDEs, from scratch. Deliver
it as a portable package (`bash infer.sh INPUT_DIR OUTPUT_DIR`). An independent verifier replays
it on 1,085 held-out trajectories and scores the free-running rollout.

**Why it is hard.** The four sources share nothing:

- their equations differ;
- channel counts range from 1 to 11;
- grids are 256², 128×384 and 128²;
- boundary conditions differ.

One model has to handle all of them from four context frames, then roll forward on its own output
for up to 30 steps. Nothing about the dataset is supplied at inference beyond the frames and the
simulation's physical constants.

**The scientific question.** Can one model learn many equations and transfer to unseen physics?
And what do architecture, objectives and self-run pretraining actually buy at small scale? This
is the question the foundation-models-for-PDEs line of work poses (PDE-Transformer, Poseidon,
DPOT, MPP). The question is not new; the measurement is:

- No released checkpoint leads on every group under this protocol.
- None transfers zero-shot to physics it was not trained on.
- An 18.6M model trained from scratch on this task's data beats every released checkpoint on two
  of four groups: by 2× on the turbulent layer and 2.5× on PDEArena (4× against the released
  PDE-Transformer checkpoints alone).

**The bar.** The authors' released PDE-Transformer checkpoints, replayed under this protocol, are
the external target. Any architecture is allowed. No pretrained weights are supplied or
permitted, so an agent starts from random initialization.

**Method landscape.**

- **Channel-shared U-Net** (the starter baseline): no cross-channel coupling, no conditioning.
- **PDE-Transformer** (Holzschuh et al., ICML 2025): a windowed-attention multi-scale transformer
  with PDE conditioning. It is the external target.
- **FNO / TFNO, U-Net, CNextU-net** (The Well baselines): per-dataset models. CNextU-net, the
  strongest, is a verified baseline here.
- **DPOT** (Hao et al., ICML 2024): an AFNO transformer pretrained on twelve datasets. It
  collapses on unseen physics under this protocol.
- **Poseidon, MPP, BCAT, PROSE-FD**: audited in `environment/references/SOTA_AUDIT.md`, and
  excluded for license, data-availability or protocol reasons.

**Status.**

- Data pinned, verified on real downloads, and published as one pinned release.
- The trivial predictors, the starter and CNextU-net are reproduced by the integrator
  (section 6).
- Inference runs isolated from the held-out targets. A red-team probe proved the isolation.
- The workspace follows [TASK_STANDARD.md](../TASK_STANDARD.md).

## 2. Data

The data is published as one pinned release,
[deqing/ai4sci-pde-surrogate-data](https://huggingface.co/datasets/deqing/ai4sci-pde-surrogate-data)
@ `f99c579e` (release `pde-surrogate-split-v2`).

- The release carries the 39.32 GB this split creates: the truncated training heads and the
  held-out tails of the nine corpus files the split cuts.
- The rest comes from the four upstream sources below. Each is pinned to a Hugging Face commit
  and downloaded file by file, with checksum and trajectory-count verification
  (`environment/data/MATERIALIZE.md`). Every downloaded file is recorded in
  `docs/validation/materialize-receipt-2026-09-16.json`.

| source | what | license | held-out cases | K |
| --- | --- | --- | ---: | ---: |
| `thuerey-group/pde-transformer-ape2d-full` | pretraining corpus: 16 datasets from 9 equations (Gray-Scott in 8 regimes), 256², 600 trajectories each | MIT | 850 (+270 long-horizon) | 26 |
| `polymathic-ai/turbulent_radiative_layer_2D` | compressible radiatively cooling mixing layer, 128×384 | CC-BY-4.0 | 9 | 30 |
| `polymathic-ai/active_matter` | active nematic continuum, 11 channels, 256² | CC-BY-4.0 | 26 | 30 |
| `pdearena/NavierStokes-2D` | incompressible buoyant smoke, wall-bounded, 128², 14 steps | MIT | 200 | 10 |

**Split.** `environment/data/split.json` is the task's own trajectory-level split:

- 10,115 training and 585 validation trajectories, and 1,085 primary held-out cases;
- generated deterministically from the lock and tested for leakage.

The paper's split cannot be reproduced from public artifacts:

- the corpus exists in two releases that share parameters but not data;
- the public loader is hard-coded to the small one;
- test sets are defined differently per PDE;
- the paper's transfer subsets from The Well were never published.

The full release's `hyp.hdf5` is excluded: it is a content copy of `diff.hdf5`
(`environment/data/AUDIT.md` §7).

**Materialization (controller).** Harbor does not run this step.

```bash
# agent-side development data (138 GB): the release's train/ subtree plus the upstream training pools
export TASK_DATA_DIR=/large-disk/pde-surrogate-agent
python3 pde-surrogate/environment/data/materialize.py "$TASK_DATA_DIR" --from-release --subtree train
python3 pde-surrogate/environment/data/materialize.py "$TASK_DATA_DIR" --pools pde-surrogate/environment/data/split.json --sources -
# verifier-side private data (63 GB): the 54 files the held-out cases reference
export TASK_PRIVATE_DIR=/large-disk/pde-surrogate
python3 pde-surrogate/environment/data/materialize.py "$TASK_PRIVATE_DIR" --manifest pde-surrogate/environment/data/split.json --sources -
```

`TASK_DATA_DIR` then holds:

- `ape2d-full/`, `pdearena-ns/`, `well-active_matter/` and `well-turbulent_radiative_layer_2D/`;
- `release-receipt.json` and `materialize-receipt.json`.

Each entry is mounted read-only at `/workspace/data/<entry>`. The development entrypoint refuses a
corpus file that still carries a held-out tail.

## 3. Workspace

| path | contents |
| --- | --- |
| `/workspace/instruction.md` | the prompt |
| `/workspace/contract.md` | `pde-surrogate-rollout/v1`: package, I/O schema, runtime and time limit, scoring, rules |
| `/workspace/data/` | `README.md`, `AUDIT.md`, `split.json`, `split_smoke.json`, `sources.lock.json`; the four datasets (read-only mounts) |
| `/workspace/baseline/starter/` | channel-shared U-Net, 473k parameters: train, publish (atomic), infer; reproduced reward **0.3127** |
| `/workspace/baseline/cnext_unet/` | The Well's CNextU-net recipe, 18.6M, from scratch on PDEArena, with The Well's source at `/opt/the_well`; reproduced reward **0.3447** |
| `/workspace/references/` | released-model replays (`released_models.md`) and the task designer's audit (`SOTA_AUDIT.md`), not verified by the integrator |
| `/workspace/evaluation/` | the verifier's scorer, the case builders and a tiny synthetic contract example |
| `/workspace/submission/` | empty; the agent keeps the current package here |
| `/workspace/research/` | empty, writable, on a large disk (`TASK_RESEARCH_DIR`) |

The workspace holds no network, no held-out targets and no released weights. The replay packages
for released checkpoints stay with the maintainers in `docs/validation/released_models/`.

## 4. Evaluation

**Protocol.**

- Each held-out trajectory gets four context frames and a free-running rollout from its start.
- Rollouts are K = 26 steps for the corpus, 30 for The Well and 10 for PDEArena.
- Per channel and step, VRMSE = sqrt(MSE / (Var(target) + 1e-7)), with the unbiased spatial
  variance, clamped at 10. It is averaged over channels, steps and cases per dataset.
- The corpus counts as one group and each transfer dataset as its own.
- The reward is the mean over groups of a log-band map: VRMSE 0.01 → 1.0, 0.1 → 0.667,
  1.0 → 0.333, ≥ 10 → 0.
- The grader equals The Well's reference implementation to machine precision. Failed cases stay
  in the denominator.

Full definition: `environment/contract.md`. Implementation: `environment/evaluation/score.py`.

**Verifier.** `tests/` builds a separate offline image. The private data sits under a root-only
`/private`. The package runs in a Bubblewrap namespace or, where Docker refuses one, as uid 10001.
It can then read neither the truth nor `/tests`, and `reward.json` records the mode. It
writes `/logs/verifier/reward.json`:

- `reward`: the task reward above;
- `normalised_score`: 0 for persistence and 1 for the per-group oracle
  (`environment/evaluation/score_bounds.json`);
- per-group details in `report.json`.

## 5. Running

```bash
export TASK_DATA_DIR=... TASK_PRIVATE_DIR=...                  # section 2
export TASK_RESEARCH_DIR=/large-disk/runs/pde-surrogate-$(date +%s); mkdir -p "$TASK_RESEARCH_DIR"
harbor trial start -p pde-surrogate -a <agent> -m <model> --allow-agent-host <model API host>
```

The external budget is in `task.toml`: 48 hours, eight H100s, at least 64 CPU cores, 512 GiB RAM and
1 TiB storage for development. The verifier gets one GPU, four hours for inference, and six
hours in total. Measured costs:

- the starter trains 30,000 steps in 24 minutes on an H100 (40 on an A100);
- CNextU-net needs 10.1 h for 200 epochs on an H100;
- replaying all cases takes 65 s for the starter on an H100, and 27–47 minutes for the released
  checkpoints on an A100.

Maintainer commands (tests, reference replays, release rebuild, image builds) are in
[`docs/RUNNING.md`](docs/RUNNING.md). The unit tests run with `bash tests/run_unit_tests.sh`.

Repository layout:

| path | contents |
| --- | --- |
| `instruction.md`, `task.toml` | the prompt; Harbor configuration |
| `environment/` | development image: `Dockerfile`, `docker-compose.yaml`, `contract.md`, `data/` (docs, split, lock; controller-side materializer, release builder and publisher, split generator), `baseline/`, `references/`, `evaluation/`, `entrypoint.sh` |
| `tests/` | `test.sh`, sealed replay driver, isolation module, compose overlay, Dockerfile, unit tests |
| `docs/validation/` | receipts and replay records; `released_models/` holds the maintainer-only replay packages for released checkpoints and the fine-tune tooling |
| `docs/` | `RUNNING.md`, `BENCHMARK_READINESS.md`, `AGENT_EVALUATION.md`, `UPSTREAM_LICENSES.md`, `compute-profiles.toml`, figures |

## 6. Results and validation

Table 1 holds only numbers produced on this task's held-out cases, with this task's grader, under
the protocol above. Only these may be compared with each other or with an agent's score.

- The task author's rows are single runs on a shared A100-40GB (`docs/validation/`), recomputed
  exactly for the 1,085-case split (`docs/validation/drop_duplicate_family.py`).
- The integrator's reproductions (one H100, the task's images, the isolated verifier) are in the
  last column.

**Table 1. VRMSE under the task protocol.**

| model | corpus (850) | turbulent layer (9) | active matter (26) | PDEArena NS (200) | reward (author) | reward (integrator) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial mean of the last frame | 1.25 | 1.51 | 1.00 | 1.00 | 0.3105 | 0.3105 |
| persistence | 1.19 | 1.36 | 1.52 | 1.65 | 0.2827 | 0.2827 |
| **baseline: starter**, channel-shared U-Net, 30,000 steps | 1.15 | 1.11 | 1.54 | 0.87 | 0.3139 | 0.3127 |
| **baseline: CNextU-net** (The Well's recipe), PDEArena model, best validation to epoch 120 | — | — | — | 0.30 | 0.3446 | 0.3447 |
| external target: released PDE-Transformer `mc-s`, zero-shot | 0.42 | 1.40 | 1.32 | 1.22 | | |
| external target: released PDE-Transformer `mc-b`, zero-shot | 0.38 | 1.33 | 1.32 | 1.36 | | |
| external target: released PDE-Transformer `mc-l`, zero-shot | 0.34 | 1.44 | 1.53 | 1.38 | | |
| external: released DPOT-Ti, zero-shot, context padded 4 → 10 frames | 1.72 | 7.77 | 2.19 | 0.75 | | |
| external: released DPOT-M, zero-shot, context padded 4 → 10 frames | 1.63 | 7.60 | 2.08 | 0.96 | | |
| from scratch: CNextU-net, one model per transfer dataset, 12 h each | — | 0.67 | 1.89 | 0.30 | | |
| external: PDE-Transformer `sc-s` fine-tuned per transfer dataset (1 input frame) | — | 1.17 | 1.67 | 0.76 | | |
| external: the same `sc-s` fine-tuned on all three transfer pools at once | — | 2.01 | 1.81 | 1.29 | | |

- **External rows** use released weights that agents do not get.
- **Dashes** mark per-dataset recipes. Their packages persist on the other groups, so the
  whole-package rewards are:
  - CNextU-net: 0.308, 0.275, 0.345;
  - fine-tunes: 0.288, 0.279, 0.311; the joint fine-tune 0.271.
- **The CNextU-net rows** ran within The Well's 12-hour budget on an A100: 200 epochs on the
  turbulent layer, 123 on PDEArena and 29 on active matter.
- **The integrator's CNextU-net re-run** completed all 200 epochs in 10.1 h on an H100. Its best
  validation over all 200 epochs scores 0.3466 (PDEArena VRMSE 0.282).
- **The author's starter run** indexed 19 of 20 training datasets and 2 of 52 PDEArena shards.
  The loader at the time skipped missing pools silently; it now refuses them
  (`docs/validation/task_trained_baseline.json`).

![Every measured system placed between each track's trivial floor and its ceiling, on a normalised scale](docs/score_landscape.png)

*Both tracks are rescaled so the trivial floor reads 0 and the ceiling reads 1
(`docs/validation/score_bounds.json`; regenerate with `python3 docs/make_score_landscape.py`). The
PDE ceiling is an oracle the contract forbids a submission from reaching. No legal single package
has passed +0.45.*

**Table 2. Other protocols; not comparable with Table 1 or with each other.**

| source | protocol | number |
| --- | --- | --- |
| released DPOT-Ti / DPOT-M, native context | 10 context frames, K = 4, the same 200 PDEArena trajectories | 0.14 / 0.09 |
| released `mc-s` on this task's long-horizon set | 4 context frames, K = 96, the 270 long-rollout test trajectories of 7 corpus PDEs | 0.91 |
| The Well, published baselines | one model per dataset, 4 context frames, 12 h on one H100, VRMSE over rollout steps 6–12 / 13–30 | turbulent layer: FNO 1.79 / 3.54, U-net 0.66 / 1.04, CNextU-net 0.54 / 1.01 |
| PDE-Transformer paper | 1 context frame, nRMSE in per-dataset normalized units, mean over the corpus | `mc-s` 0.044 after 1 step, 0.36 after 10 |

The readiness record, the review findings and their fixes are in
[`docs/BENCHMARK_READINESS.md`](docs/BENCHMARK_READINESS.md) and `docs/release_validation.json`.

## 7. Limitations

- The frozen score uses the four pinned sources and this task's split.
- Four context frames follow The Well. This handicaps ten-frame designs (Table 2).
- The per-group reward weights the sixteen-dataset corpus equally with each transfer dataset.
- All upstream data are public, so this is a benchmark-visible evaluation, not a secret test set.
- Inference timings for the released checkpoints were measured on an A100.

## 8. Citation and licenses

Upstream licenses: [`docs/UPSTREAM_LICENSES.md`](docs/UPSTREAM_LICENSES.md) (MIT and CC-BY-4.0
data; BSD-3 `the_well`).

- Holzschuh, Liu, Kohl, Thuerey. PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations. ICML 2025. arXiv:2505.24717.
- Ohana et al. The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning. NeurIPS 2024 Datasets and Benchmarks. arXiv:2412.00568.
- Gupta, Brandstetter. Towards Multi-spatiotemporal-scale Generalized PDE Modeling (PDEArena). 2022. arXiv:2209.15616.
- Hao et al. DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training. ICML 2024. arXiv:2403.03542.
- Koehler et al. APEBench: A Benchmark for Autoregressive Neural Emulators of PDEs. NeurIPS 2024 Datasets and Benchmarks.
- Herde et al. Poseidon: Efficient Foundation Models for PDEs. 2024. McCabe et al. Multiple Physics Pretraining. 2023. Li et al. Fourier Neural Operator. ICLR 2021.
