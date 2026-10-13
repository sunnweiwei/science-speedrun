# Data audit for the PDE surrogate task

Status: audit of the pinned sources, written before any task code and kept as
the record of what was found. Every number below was read from the Hugging Face
API at the pinned revision, from a downloaded file, or from the upstream
repository at its current `main`; the source is named in each section. Items
marked **unverified** in the sections below were verified on real downloads on
2026-09-16 (`docs/validation/materialize-receipt-2026-09-16.json`); §7 (2026-09-22)
records the one defect the receipt could not see.

## 1. What the task needs from data

An agent develops one autoregressive surrogate for 2D time-dependent PDEs on
regular grids. The verifier holds out trajectories, gives the package the
initial state and metadata, and scores the K-step rollout. Data must therefore
be (a) pinned to an immutable revision, (b) split by trajectory with the split
published as an artifact, (c) small enough to materialize selectively, and
(d) documented well enough that another person can rebuild the same files.

## 2. Pinned sources (`sources.lock.json`)

`build_sources_lock.py` regenerates the lock from the Hugging Face API. Each
HDF5 file carries its byte size, upstream LFS SHA-256, split, and a trajectory
count **inferred from the byte size**; `materialize.py` reads the true count
from each downloaded file and records it in a receipt.

| id | repository @ revision | license | HDF5 files | bytes | trajectories (inferred) |
| --- | --- | --- | ---: | ---: | --- |
| `ape2d` | `thuerey-group/pde-transformer-ape2d` @ `abd0a02d` | MIT | 27 | 9.28 GB | train 800, test 27 |
| `ape2d-full` | `thuerey-group/pde-transformer-ape2d-full` @ `fa711d56` | MIT | 24 | 78.65 GB | train 6,200, test 270 |
| `well-active_matter` | `polymathic-ai/active_matter` @ `dc3e9135` | CC-BY-4.0 | 82 | 55.81 GB | train 175, valid 24, test 26 |
| `well-turbulent_radiative_layer_2D` | `polymathic-ai/turbulent_radiative_layer_2D` @ `2ee77565` | CC-BY-4.0 | 27 | 7.85 GB | train 72, valid 9, test 9 |
| `pdearena-ns` | `pdearena/NavierStokes-2D` (pinned in the lock) | MIT | 156 shards | 42.96 GB | 100 per shard: 52 train, 8 valid and 8 test shards used |
| `well-rayleigh_benard` | `polymathic-ai/rayleigh_benard` @ `10e143eb` | CC-BY-4.0 | 105 | 369.94 GB | train 1,400, valid 175, test 175 |
| `well-shear_flow` | `polymathic-ai/shear_flow` @ `fc867f85` | CC-BY-4.0 | 84 | 471.17 GB | train 896, valid 112, test 112 |

The `ape2d` byte count is after de-duplication: the reduced release stores
every file two or three times under different paths with identical SHA-256
(`adv.hdf5`, `adv/adv.hdf5`; `gs_alpha_test.hdf5`, `gs_alpha/gs_alpha_test.hdf5`,
`gs_alpha_test/gs_alpha_test.hdf5`). A naive `hf download` of the repository
fetches 18.7 GB for 9.3 GB of distinct data. The lock keeps the
`<name>/<name>.hdf5` layout, which is the one with a `meta_all.json` beside it.

## 3. The pretraining corpus (PDE-Transformer, ICML 2025)

Facts, with sources:

- **Layout** (downloaded `gs_alpha_test.hdf5`, `decay_turb_test.hdf5`): one
  file per PDE, group `sims` with attributes (`PDE`, `Fields`, `Fields Scheme`,
  `Resolution`, `Time Steps`, `Dt`, `Boundary Conditions`, `Constants`, ...),
  datasets `sims/sim<i>` of shape `(T, C, 256, 256)` float32 with per-simulation
  constant attributes, plus precomputed `norm_*` buffers at the top level.
  This is the PBDL-dataset format documented in
  `pdetransformer/data/DataModulesAndDatasetStructure.md`.
- **Paper protocol** (arXiv 2505.24717 §4.1, Appendix C): 16 PDE types, 600
  trajectories of 30 steps each at 256², generated at 2048² and downsampled;
  fixed train/validation/test split. Gray-Scott appears as 8 parameter
  variants with 100 trajectories each.
- **Two releases.** `ape2d-full` (uploaded 2026-05-17) holds the paper's
  counts: 600 per PDE, 100 per Gray-Scott variant, 50 per long-rollout test
  file, 30 per Gray-Scott test file. `ape2d` (2025-06-15) holds 60 / 10 / 5 / 3.
  The reduced release is **not** a subset of the full one. Its simulations carry
  the same per-simulation constants and seeds as the first simulations of the
  full files, but the arrays are different draws (bytewise comparison of every
  comparable file fails; correlations between same-index simulations are near
  zero or negative; `docs/validation/reduced_vs_full.json`). The two releases are
  therefore separate datasets that share a parameter list, and the smoke split
  built on the reduced release is plumbing only.
- **The public loader targets the reduced release.**
  `pbdl_datatypes/ape_2d_xxl.py` hard-codes `train_sims = range(0, 50)`,
  `test_sims = range(50, 60)` for nine PDEs and `range(0, 80)` / `range(80, 100)`
  for `gs_delta/theta/iota/kappa`, with the full-size `500/600` lines commented
  out. Two consequences: the code does not reproduce the paper's split on the
  full release without editing, and the `80/100` split for the four Gray-Scott
  variants cannot run on the reduced release, which has 10 trajectories.
- **Test-set definition differs by PDE** (same file). Seven PDEs
  (`ks`, `decay_turb`, `kolm_flow`, `gs_alpha/beta/gamma/epsilon`) have separate
  `_test` files with 200 or 100 steps; the other ten are split by simulation
  index inside the training file. Validation is a `random_split(0.85/0.15)`
  with seed 46 inside the code, not a data artifact.
- **Three PDEs are not in the training config.** `adv`, `adv_diff`, `disp` exist
  only in the reduced release; `config/ape_2d/data/multi_task_norm.yaml`
  lists 17 names, and the full release ships exactly those 17 plus 7 test
  files. The paper's "16 PDEs" counts Gray-Scott once.
- **Generator** (`simulations_apebench/`): `exponax==0.1.0` on JAX, with the
  per-PDE parameter ranges in `simulation_setups_2d.py`. Fresh held-out
  instances can be generated from pinned seeds; **unverified** here.

### Metric as implemented upstream

`pdetransformer/metric/mse.py::SimulationMSE(normalized=True)`: per
trajectory and rollout step, `sqrt(mean_{c,x,y}(pred - target)^2 /
mean_{c,x,y}(target^2))`, then averaged over trajectories per dataset; the
paper's Table values average the steps. The input frame is frame 0 and the
rollout is 29 steps (`test_unrolling_steps: 29`; for test files the loader
trims so that exactly one window per trajectory exists). Predictions and
targets are compared in the **per-dataset mean-std normalized space** set by
`normalize_data: 'mean-std'` in the data module, not in physical units. Any
task metric must state which space it uses.

### Released checkpoints (`thuerey-group/pde-transformer` @ `d5889aab`)

Six supervised checkpoints, MIT: `mc-s` 133 MB, `mc-b` 522 MB, `mc-l` 2.07 GB,
`sc-s` 187 MB, `sc-b` 716 MB, `sc-l` 2.81 GB; 100 epochs on the pretraining
corpus. Documented nRMSE after 1 / 10 steps: SC-S 0.043 / 0.34, SC-B 0.037 /
0.29, SC-L 0.034 / 0.26, MC-S 0.044 / 0.36, MC-B 0.038 / 0.31, MC-L 0.034 /
0.27 (`docs/pretrained-weights.md`). Whether these were trained on the reduced
or the full release is **not stated** upstream; the June 2025 upload date
precedes the full release's May 2026 upload.

## 4. The transfer datasets (The Well)

- **Layout** (downloaded `turbulent_radiative_layer_tcool_0.03.hdf5`,
  `active_matter_L_10.0_zeta_1.0_alpha_-2.0.hdf5`): one file per physical
  parameter combination, root attributes `n_trajectories`,
  `simulation_parameters`; groups `t0_fields/<name>` `(n, T, X, Y)`,
  `t1_fields/<name>` `(n, T, X, Y, 2)`, `t2_fields/<name>` `(n, T, X, Y, 2, 2)`;
  `boundary_conditions/<name>/mask`, `dimensions/{time,x,y}`, `scalars`.
  Files are contiguous float32 but carry 18–40% unallocated space, so byte
  size only bounds the trajectory count.
- **Per-file trajectory counts vary**: 1–4 for active matter, 40 per training
  file for Rayleigh-Bénard, 32 for shear flow. Selecting trajectories means
  downloading whole files of up to 13 GB.
- **The paper's transfer protocol is not reproducible from public artifacts.**
  Appendix E.1 says 42 / 8 / 10 trajectories were "randomly selected" from
  each split and "truncated randomly to 30 frames", with orientation and
  strain channels dropped for active matter. The public code
  (`download_well_2d.py`) streams from the Hub, subsamples with
  `torch.manual_seed(reduce_seed)` and a shuffled `DataLoader`, and writes its
  own HDF5; `pbdl_datatypes/well_2d.py` then consumes whole `_train/_valid/_test`
  files. No public code selects 42/8/10 or truncates to 30 frames (GitHub
  code search for `sel_sims` and `trim_start` finds only the loader itself).
  The reference numbers (Table 7: PDE-S pre-trained 0.455 / 0.104 / 0.125
  nRMSE on active matter / RBC / shear flow) therefore cannot be matched to a
  fixed test set. This task must publish its own trajectory-level split.
- The Well card files (`<name>.yaml`) do not describe every file: the
  `shear_flow` card lists 4 trajectories per file while its training files hold
  32. They are not usable as a manifest.

- **Which release the released checkpoints were trained on is unknown, and
  the two releases are different draws.** The reduced release's in-file "test"
  simulations (indices 50–59) share parameters with full-release training
  simulations but not data. The task's canonical split uses the full release's
  indices 500–599 and the separate `_test` files, matching the paper's stated
  ranges; the smoke split is for plumbing only.

## 5. Decisions taken for this task

1. Pin every source to the full commit SHA above; never download a repository
   wholesale; materialize by explicit file list from the lock.
2. Use `ape2d-full` as the pretraining corpus (paper counts, 30–50 test
   trajectories per long-rollout PDE) and the `ape2d` prefix as the smoke slice.
3. Publish the transfer split as an artifact: a manifest naming each Well file
   and trajectory index for train / validation / held-out test, with SHA-256
   of the converted arrays, generated by a deterministic converter (no
   streaming, no torch shuffle).
4. Define the metric once, in this task's own grader, and state the
   normalization space; treat upstream numbers as context, not as directly
   comparable references until reproduced under this protocol.

## 6. Open items

- Real downloads of `ape2d-full` and the selected Well files with SHA-256
  verification (only four files have been downloaded so far).
- Confirmation of which corpus the released checkpoints were trained on.
- Whether additional benchmark sources (PDEBench, PDEArena, further Well
  datasets) are added; a separate audit of those is in progress.

## 7. Corpus duplicate found on 2026-09-22

Fingerprinting every HDF5 of the full release by content (the `PDE` attribute,
the constants and the SHA-256 of the first and last trajectory arrays) shows
that `hyp.hdf5` is not hyper-diffusion: its `sims` group carries
`PDE = "Diffusion"`, the constants `Viscosity X` / `Viscosity Y`, the same
per-simulation values and seeds as `diff.hdf5`, identical arrays for every
trajectory compared, identical `norm_*` buffers and a byte-identical sidecar
`hyp.json == diff.json`; only the HDF5 container bytes differ, which is why the
per-file SHA-256 check in the receipt is blind to it. The reduced release's
`hyp/hyp.hdf5` is a genuine hyper-diffusion set (`Hyper-Diffusivity`), so the
full release's file is a mis-upload. No other pair of files shares content.

Consequences and what was done: the original split scored diffusion twice under
two names (100 of the 950 corpus held-out cases, 480 training trajectories);
`make_split.py` now excludes `hyp` from every pool and from the held-out set
(16 corpus datasets, 850 corpus cases, 1,085 canonical cases in all), the data
release was rebuilt without it (`pde-surrogate-split-v2`), and every canonical
record was recomputed exactly from its per-dataset means
(`docs/validation/drop_duplicate_family.py`). Released PDE-Transformer checkpoints
scored the two copies within 0.0002 VRMSE of each other, the difference being
the wrong family label they were conditioned on; DPOT scored them identically.
Reported upstream. The receipt now records a per-file content fingerprint only
through this audit; `tests/test_task.py` asserts the split never uses the file.
