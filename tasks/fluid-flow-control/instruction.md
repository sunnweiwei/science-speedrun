# Continuously improve one HydroGym controller

## Task and goal

Research **one learned closed-loop flow-control model** and improve its performance across the
complete, fixed HydroGym task suite defined by `/workspace/evaluation/case_spec.py`, using one
jointly trained parameter set for every case. The release (`hydrogym-public-research-v2`) fixes the
training/test membership:

- exactly 39 variants, 20 in 2D and 19 in 3D;
- one copy of fixed source data within a 500 GB data budget.

Your model must serve all 39. The verifier scores 26 of them, so that a replay fits its time budget:
every 2D case and one 3D case per family (`VERIFIER_CASES` in `case_spec.py`; every family, dimension
and solver backend stays scored).

That budget does not limit training interactions, generated experience, model checkpoints or
runtime storage. This is the selected public-resource version, not the original full paper suite.

Optimize each case's frozen primary metric in `/workspace/evaluation/METRICS.md`:

- CD;
- L/D;
- load or angle error;
- TKE;
- wall shear.

The task score (`score` and `reward` in `reward.json`) aggregates the 26 verifier cases by their
**10 flow families** (`case.split("_")[0]`: `Channel`, `Cube`, `Cylinder`, `Kolmogorov`, `NACA0012`,
`NACA0012Gust`, `Nozzle2D`, `Pinball`, `Sphere`, `SquareCylinder`), each weighted equally (`1/10`):

- When multiple evaluation seeds are scheduled per case, the case's physical metric `val` is the
  equal-weight mean across seeds; any diverged, failed, or invalid-operating-regime episode fails
  the case (`gain = -1.0`).
- Per case against its paired zero-control anchor `z`:
  `gain = clip(((val - z) / |z|) * (-1 if lower_is_better else +1), -1.0, 1.0)` (failed or
  nonfinite case -> `-1.0`).
- Family mean: `family_gain = mean(gain for case in family)`.
- Overall `signed_score = (1/10) * sum_{f=1}^{10} family_gain`, and final task
  `score = clip(signed_score, 0.0, 1.0)`.
- It also reports `signed_score`, per-family means, per-case gains, and each original physical
  measurement. Preserve action bounds and report stability and control cost alongside performance.

## Workspace

- `/workspace/contract.md`: the model package and inference contract.
- `/workspace/data/`:
  - `README.md`, the release descriptors (membership, source hashes, configurations, selection) and
    the release code the evaluator imports;
  - `release/` is the prepared training and evaluation resources, read-only (also reachable as
    `/data`).
- `/workspace/baseline/starter/`: shared PPO. It covers native training and resume of one
  parameter file across cases, atomic publication, inference and random initialization.
- `/workspace/references/`:
  - `specialists/`: the verified per-case PPO/DDPG/TD3 trainer, export and recipes;
  - `results/`: 39/39 specialist results, next to uncontrolled and paper references.
- `/workspace/evaluation/`:
  - `README.md`, `METRICS.md` and `case_spec.py`;
  - the self-service native evaluator (`harbor_verify.py`, `evaluate_native.sh`).
- `/workspace/research/`: writable space for runs, experience, checkpoints and notes.
- `/opt/hydrogym` and `/opt/references/`: upstream physics, the authors' model/loss source, and an
  optional world-model architecture reference.

## Rules

- This is **known-task multi-task reinforcement learning**, not a claim of generalization to unseen
  tasks.
  - You may interact with and train on every task configuration in the suite, including its
    supplied initial-condition pool.
  - Observations, actions, rewards and the resulting trajectories are legitimate training data.
  - No separate external fluid dataset or new task family is provided. Excluded or unrelated
    archive cases are not additional training data.
- The suite must bind exactly this membership. Never redefine it based on which cases your model
  handles well.
- The Channel 3D runtime is `channel-72cube-local-actuator-clock-v1`, in
  `/workspace/evaluation/native_channel.py`. Use this frozen adapter: the upstream class alone has
  a documented actuator-clock defect.
- Three 3D MAIA configurations (`Cube_3D_Re3700` in the 26-case verifier suite, plus
  `Cylinder_3D_Re3900` and `NACA0012Gust_3D_Re50000_AOA12` in the 39-case development suite) launch
  two MPI solver ranks (`solver_processes = 2`). When two or more GPUs are visible,
  `maia_gpu_rank.py` binds each MAIA solver rank to a distinct GPU (`devices[rank - 1]`). On a
  single 80 GB H100 (`HYDROGYM_ALLOW_SHARED_GPU=1`, enabled by default; set `"0"` to enforce strict
  distinct-GPU binding), both MAIA solver ranks cleanly share the single GPU
  (`devices[(rank - 1) % len(devices)]`) under `mpirun --oversubscribe --bind-to none` (compatible
  with CUDA MPS) and container-safe OpenMPI BTL settings (`OMPI_MCA_btl=self,vader,tcp`,
  `OMPI_MCA_btl_vader_single_copy_mechanism=none`).
- Development replay (`case_spec.py`) uses `seeds=[123]` by default, whereas sealed blind
  evaluation replays multiple private perturbation seeds per case and masks the secret simulator
  seed in `policy.reset(...)` (passing only the public `episode_index`, e.g. `0, 1, 2`). Controllers
  must rely on closed-loop observation feedback rather than open-loop step memorization.
- Preserve the evaluator's:
  - geometry, solver fidelity, observation locations and actuators;
  - physical action bounds, control interval, reward and episode horizon.
- You may change architecture, representation, optimization, experience sampling, training schedule
  and inference computation. Surrogate or world models may be learned from allowed interactions.
  Their usefulness is measured in the original CFD environment, not by their own predicted reward.
- All task specifications, reference results and evaluation code are visible. Run development
  evaluation directly and use it for analysis and global model selection.
  - During an evaluation replay, model weights and normalization statistics are fixed. Do not turn
    the scoring replay into a training or adaptation run.
  - Record development feedback use in the submission README; this is not a blind test.
- Submit one logical checkpoint containing one model's learned parameters.
  - Do not train or select independent per-case controllers and bundle them behind a dispatcher,
    even in a single file.
  - Allowed: shared task-conditioned components, jointly trained input-dependent computation, and
    a single jointly trained world model plus its controller or planner.
- There is no parameter-count constraint; choose capacity based on measured learning and compute
  requirements. No pretrained candidate weights are supplied. Do not use unrelated data or weights.
- There is no network.

## Submission

Maintain `/workspace/submission` as the current complete model package described in the contract.
Publish a trained, replayable candidate early and update it throughout research. The package may be
read at any moment; the starter's publication replaces it atomically. Keep experiment archives,
optimizer state and old models under `/workspace/research`, outside the package. A zero-action
diagnostic is not a trained candidate or the main research line.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

The per-case reference values (the 39 specialists and each case's uncontrolled value) are in
`/workspace/references/results`, as `baseline/starter/RESULTS.md` explains: compare candidates with those
numbers instead of replaying the baselines. Start by checking your own pipeline cheaply: one real training update,
a publication, and a one-case closed-loop replay (`--case Cylinder_2D_Re200`, see
`/workspace/evaluation/README.md`). A complete 39-case replay is expensive (single cases take from under a
minute to over an hour of simulation), so iterate on a representative subset of cases and replay the full
suite for publication candidates. Then repeatedly:

1. analyze per-case results and formulate a testable research hypothesis;
2. implement and train a meaningful candidate using the supplied environments;
3. evaluate the same checkpoint across cases, and investigate regressions and costs;
4. publish the strongest complete replayable candidate when evidence improves;
5. immediately select and execute the next experiment.

This is continuing research, not a finite deliverable. Do not declare completion or return a final
answer after a training run, a successful publication, an empty task list or a reached score
target. Keep concise research notes with:

- fixed per-case primary metrics;
- the hypothesis;
- training provenance and CFD interaction counts;
- wall-clock costs.
