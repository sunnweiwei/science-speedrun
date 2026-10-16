# Coverage expansion under the 500 GB source-data cap

## Final outcome — 2026-09-07 22:15 UTC

The finite audit queues finished. The reviewed expansion is now frozen as
`hydrogym-public-coverage-v1`: **40 cases (21 two-dimensional, 19 three-dimensional),
576 source files / 486,282,110,078 bytes**. It preserves all first 28 cases.
All included cases have clean native reset/control/reset probes with at least
two warm samples. The [final table](COST-RESULTS-20260907.md) accounts for all
62 complete source variants and distinguishes measured high cost, source-budget
tradeoffs and unvalidated runtime/protocol bindings. This is a final cost/resource
decision, not complete physical evaluator certification or a paper score.

Correctly bound two-H100 means: Cube Re3700 34.626 s/control;
Cylinder Re3900 88.976; gust wing Re10000 38.954; gust wing Re50000 35.869.
Each has four warm samples and a completed second reset. The cube Re300 also
passed, but adding its 14.09 GB would put the selected set at 500.372 GB, above
the hard cap; the high-Re cube is retained instead. Unmeasured variants are not
labelled slow.

The notes below preserve the audit history; earlier candidate counts and
"queued/running" language are historical snapshots, not current execution status.

Organizer direction, 2026-09-07: maximize worthwhile, relatively affordable
task coverage within 500 GB. The 28-case / 138.27 GB `speed-v1` release is a
preserved first tested set, **not evidence that all other cases are slow or
the final coverage decision**. Do not mutate that version's data/evidence.

The broader complete-resource inventory has 62 variants / 1,093.33 GB.
Of the 34 excluded variants, only four had measured costs over the organizer's
old 30-second cutoff (98.08 GB). Six were initialization-limited (250.24 GB),
eight lacked effective speed measurements (442.93 GB), and sixteen had
configuration, runtime or protocol issues (163.81 GB). None of the latter
groups is evidence of a measured slow solver.

The old 30-second threshold is not a paper rule or user requirement and does
not determine the expanded selection. Review scientific contribution, native
episode/physical-time workload, measured cost, startup/reset amortization and
source bytes together. Do not add redundant expensive variants just to fill
space, but do not leave valuable unmeasured cases excluded as if tested slow.

## Active bounded audit

The independent audit directory on the development host is
`/mnt/workspace/hydrogym/reproduction/coverage-expansion-20260907`.
Six available H100s (0,1,2,3,4,6 at launch) serve a finite queue of sixteen
MAIA 3D candidates: all three cylinders; both higher-Re gust wings and the
Re1000/AOA40 gust wing; the six no-gust Re100/1000/10000/50000 wing variants;
Cube Re300; and all three square cylinders. Source grids, restarts, reward
implementations, time steps and native control intervals are unchanged.

The probes request five controls, including bounded nonzero actions, followed
by a second reset. Cold initialization has a separate 45-minute diagnostic
allowance, controls 15 minutes and second reset 15 minutes, within a 75-minute
outer bound. These finite diagnostic budgets are **not** model-development
timeouts or membership thresholds. A censored run remains explicitly unknown.

Short-loop rates, CFD substeps per action, native horizons, initialization,
reset, process exit, source/configuration identity and raw samples must be
reported. Episode cost obtained by multiplying a short rate by a native horizon
must be labelled an extrapolation, not an executed full-episode measurement.

Other work examines the official Kolmogorov `minimize_tke` example and the 3D
nozzle's missing actuator-count binding. Any organizer-defined configuration
must be explicit, not labelled as a reproduced paper setup. No unresolved
configuration is silently replaced, and existing specialist training is not
stopped or retargeted.

A finite follow-up may use two solver ranks on two H100s for the two higher-Re
gust wings and Cylinder Re3900 if their single-GPU probes do not succeed. It
was rebalanced before launch onto free GPUs 0/4 after the original single-GPU
attempts finished. Only the two waiting audit controllers were stopped; no
CFD/training process was interrupted. Availability is checked before launch.
The native example exposes solver rank count as a runtime choice. Successful
two-GPU timing must report two-GPU cost, not be compared as one-H100 timing;
neither memory exhaustion nor slower integration is inferred from an unexplained
native abort. The two-GPU attempt retains the same grid, restart and CFD interval.

An additional native two-GPU Cube Re3700 check uses GPUs 1/6 after the nozzle
queue. It is a distinct high-Re 3D geometry, not a duplicate of a completed
single-GPU result. These are bounded diagnostic queues, not new agent sessions.

### Explicit nozzle resource-loader correction

The pinned `Nozzle2D_fine` and `Nozzle3D_coarse` case JSONs omit `restart`.
The supplied JAX-Fluids initializer checks `case_setup.restart_setup.is_restart`
before reading even an explicitly supplied `user_restart_file_path`; without
that flag it takes the stationary primitive/zero-levelset placeholder branch.
The supplied `Nozzle2D_coarse` JSON already has the flag.

The separate `cases-restart` tests add only
`restart = {is_restart: true, file_path: <provided restart.h5>}` to the **in-memory
runtime configuration**. The source JSON/HDF5 remains unchanged, and native
grid/decomposition compatibility assertions remain enabled (no interpolation).
The small [binding helper](native_resource_bindings.py) preserves all other
fields and rejects an existing source restart choice. All eight original files
for the two corrected candidates were checksum-verified; the helper output
exactly matched both executed effective case configurations after the native
GPU x-domain decomposition setting.

With this correction, the 2D fine nozzle completed twelve controls and a second
reset at **2.918 s/control** (11 post-warmup samples), with nonzero initial flow,
changing fields and signed thrust-angle response to asymmetric 0.5 actuator
commands. It is a useful new candidate, not a permanently excluded static case.

The 3D coarse diagnostic uses **eight actuators**, an explicit organizer choice
within the source class's supported 4–12 range, and two H100s. It now passes
restart and integrates. Its first post-warmup control took **234.48 seconds**,
advancing about 4.07e-5 physical seconds; a native 0.01-second episode extrapolates
to about **16 hours wall time / 32 GPU-hours**. CFL-adaptive dt and short sampling
make this an estimate, not a completed episode or paper setup reproduction.
The finite probe subsequently completed its nonzero-control/reset check, with
two warm samples of 234.48 and 234.21 s/control and a 13.36-second second reset.

## Progress snapshot — 2026-09-07 20:52 UTC

Five new candidates have clean reset/control/reset probes with multiple warm
samples. Added to the preserved first 28, they would give **33 candidates /
233,965,928,263 source bytes (233.966 GB)**. This is **not** a frozen new list,
a stopping target, or complete evaluator certification.

| Newly measured candidate | Source GB | H100s | Warm seconds/control | Samples |
| --- | ---: | ---: | ---: | ---: |
| Cube 3D Re300 | 14.090 | 1 | 43.44 | 4 |
| Cylinder 3D Re200 | 23.164 | 1 | 42.61 | 4 |
| NACA0012 gust 3D Re1000/AOA40 | 34.949 | 1 | 20.06 | 4 |
| NACA0012 no-gust 3D Re100/AOA40 | 23.494 | 1 | 33.19 | 4 |
| Kolmogorov 2D | <0.001 | 1 | 3.46 | 6 |

Kolmogorov uses the pinned official `minimize_tke` example's `reward_alpha=1`
with class-default Re200 / 64² / float64 / dt=0.001 / 10,000 CFD steps per
action. This is explicitly not the inconsistent asset YAML or a paper-Re250
reproduction. Its objective is meaningful in this profile; full scoring
acceptance remains pending.

The single-GPU 3D coarse nozzle now reaches integration using an explicit
organizer diagnostic choice of eight actuators, but fails with an actual
JAX `RESOURCE_EXHAUSTED` GPU allocation error. A two-GPU native decomposition
probe is queued. Unexplained MAIA rank aborts remain diagnosed separately;
they are not automatically reported as OOM or measured slow.

The 2D fine nozzle completed twelve controls at ~3.02 seconds/control and a
second reset, but its primitive field remained exactly unchanged from reset
even with alternating first/last-actuator commands at 0.5. It is not counted
among the five additions: cheap execution alone does not establish useful
control. No mesh, restart, numerical precision or action interval was changed.

The audit also writes `PROGRESS.md`, `progress.json` and a review-only
`cost-evidence-coverage-candidate.json` under the audit directory. Candidate
evidence never auto-selects cases or overwrites a frozen release. The new
coverage builder/loader is unit-tested, but is not activated while membership
review remains open. Existing speed-v1 stays the default until an explicitly
reviewed expansion is generated and verified.

## MAIA GPU-binding correction (21:33 UTC)

NVML process inspection showed both MPI solver processes from the initial
two-GPU MAIA attempts on the first allocated GPU. Those two attempts
(`cases-two-gpu/Cube_3D_Re3700` and the Re10000 gust wing) were stopped and
marked **invalid_gpu_binding**, preserving their raw outputs. They cannot
qualify a case or serve as correct two-GPU cost/capacity evidence. JAX-Fluids
used both GPUs correctly and its measurements remain valid.

The replacement `cases-bound-two-gpu` probes explicitly assign a distinct CUDA
UUID to each solver rank, verify with the CUDA driver that each rank sees only
its assigned GPU, and save binding receipts. Actual MAIA processes were checked
on six distinct GPUs after launch. The two gust wings run on pairs 0/4 and 1/6;
Cube Re3700 then Cylinder Re3900 run on 2/3. Other training jobs are unaffected.

The reusable `evaluation/maia_gpu_rank.py` fixes the same multi-rank path for
future task runs. Its preflight also passed a deliberately reversed CUDA mask,
so it respects the caller's device visibility/order rather than grabbing all
GPUs advertised by NVML. Cost-selection validation rejects nominal two-GPU
MAIA evidence without distinct verified per-rank bindings.

The verbose single-GPU Cylinder Re3900 attempt reported CUDA invalid argument /
invalid device ordinal from the native allocation/Thrust path, not a confirmed
GPU out-of-memory message. This distinction is retained rather than inventing
an OOM diagnosis.

## Release gate

Freeze a separately versioned expansion only after the inclusion decisions
have measurement/provenance, its complete selected source sum is at most
500,000,000,000 bytes, and all unselected variants have honest reasons.
Materialize through the existing pinned HF/offline interface; on this machine
reuse immutable local files. No data upload is part of this audit.

One jointly trained checkpoint and frozen-parameter scoring across the complete
selected set remain unchanged. Data membership/short-loop certification is
not full scientific evaluator acceptance; complete episode and relocated-model
replay still require their own validation.
