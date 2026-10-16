# Full-suite reproduction acceptance plan

Status: **public data selection frozen; native suite not validated; not reproduced**.
Updated 2026-09-06 following the user's
explicit requirement for complete test coverage with at least one competitive
paper method, rather than exhaustive reproduction of every algorithm.

**Subsequent user decision:** use the [staged baseline plan](STAGED_BASELINES.md).
First reproduce the five cases with identified case-matched control curves;
after reviewing success, adapt the verified method to other included cases and
label those results organizer baselines. The earlier demand for a published
policy score on every resource variant is superseded by this distinction.
Data membership and the requirement for complete independent evaluation remain
unchanged. Kolmogorov is not in the initial five-case stage.

**Latest priority:** first freeze the benchmark resources/cases, delivery contract
and automated evaluation protocol. Do not make raw native evaluation conditional
on a paper baseline already being reproduced. The v1 delivery interface is now
frozen and raw-only evaluation is implemented. The user has now authorized
excluding missing resources, and subsequently approved the fixed 500 GB data scope.
[Public 500 GB release v1](../environment/data/RELEASE.md) fixes 47 resource
variants (488.77 GB) for training and test, with explicit budget and resource exclusions.
Full native binding/replay of that set remains required; adapters may not be
silently dropped. Earlier full-paper scope requirements below are superseded
only by these declared exclusions, not by a model-dependent diagnostic subset.

The subsequent [47-case evidence audit](PAPER_PROTOCOL.md), recorded in
[paper-targets.json](paper-targets.json), fixes paper locators, metric profiles,
available training defaults, uncontrolled validation values and unresolved
paper/code conflicts. It is not an executable `suite.json` or an acceptance
score table. Five cases have case-matched learning-curve candidates, Kolmogorov
has a family curve with unresolved configuration, and 41 exact variants lack a
located matching learning curve. All matched-statistic acceptance targets remain
null. Do not treat the 15 uncontrolled-flow table entries as learned-policy scores.

This plan supersedes earlier instructions to run PPO, DDPG and TD3 on every
comparison case. Existing experiments remain archived as development evidence;
they are not automatically formal training runs or accepted reference anchors.
Do not launch an AutoResearch agent before the benchmark inputs and references
are validated. No author contact, data upload or additional compute purchase is
authorized by this plan.

## Scope

Preserve the complete paper/environment scope in
[`paper-coverage.json`](../environment/data/paper-coverage.json). Neither the
seven highlighted reproduction cases nor the locally available MAIA cases are
a replacement for the selected 47-case public scope. Missing paper cases remain
visible as explicit exclusions; no claim of original full-paper coverage is made.

The full environment inventory and the paper's reported learned-policy results
are different objects. The two visually inspected paper tables contain 20 2D
parameter rows and 35 3D parameter rows. Some rows contain parameter ranges,
multiple control objectives, actuator alternatives or solver alternatives.
These 55 table rows are **not** a resolved 55-case release and do not establish
the advertised 61-environment count. Expanding the final case list still needs
explicit, source-supported choices; do not invent a Cartesian product or drop
variants merely to reach a particular count.

For every final case, distinguish:

- A published learned-controller result whose metric/setup can be recovered.
- A published result whose metric/setup still needs reconciliation.
- An environment for which a corresponding learned-controller target has not
  been located in the inspected public material.

The last category cannot be certified as a reproduced paper score by substituting
our own newly measured reference. Establishing new full-suite reference scores
is a separately labelled benchmark-construction result. The user has now
approved that distinction, conditional on first reproducing the located paper
cases and then adapting the verified method. Do not claim the later organizer
references are recovered published scores.

## 1. Freeze training and evaluation before acceptance runs

Keep one reviewed case identity across training, checkpoint provenance and test
results. The release must specify, for every case:

| Part | Required fixed information |
| --- | --- |
| Environment | Geometry, Reynolds/Mach/angle, control objective, solver and fidelity, source revision and asset hashes |
| Training resources | Allowed initial flow states and disturbances, sensor/actuator layout, physical action bounds, observation/reward preprocessing and calibration inputs |
| Training recipe | Published algorithm/code reference, documented reconstruction differences, interaction budget, independent training seeds and checkpoint-selection rule |
| Evaluation | Exact initial flow states/disturbances, control interval, episode duration and boundary semantics, frozen preprocessing, repetitions and deterministic/stochastic action convention |
| Metrics | Reward formula and coefficients, temporal/spatial aggregation, physical diagnostic definitions and windows, uncertainty reporting |
| Paper comparison | Figure/table/source, numerical target or explicitly approximate curve-derived target, matching statistic and justified comparison tolerance |

This remains the agreed **known-task training** regime. Training and evaluation
may use the same declared task configurations and initial-condition pool. No
held-out Reynolds numbers, new private flow states or outside training corpus
are introduced here. A different seed label on the same deterministic restart
does not create an independent test sample.

The training set is an allowed interaction-generating environment specification,
not a claim that a fixed offline transition dataset already exists. Learned
normalizers are fit only from allowed training data and frozen for replay.
Evaluation rollouts do not enter the learner's replay buffer or update weights.
Any development-evaluation use for checkpoint selection must be declared.

Do not publish `/data/suite.json` as validated until every included case has a
reviewed, executable binding. Missing values stay unresolved, not invented.
Existing experiment `protocol.json` files remain unchanged; a new protocol
requires separately labelled results, not relabelling old runs.

## 2. Reproduce the necessary methods, not the entire algorithm grid

First establish a competitive paper-method reproduction on the identified
stage-1 cases. Then use the verified method, with limited documented adaptation,
to build organizer reference scores across the remaining selected cases.
Select and record the target method before formal acceptance evaluation;
do not silently substitute another algorithm based on the result.

- PPO/DDPG/TD3 are the classical comparison candidates. Their appearance in a
  figure is not evidence that every one must be rerun, nor that one algorithm
  is the strongest on all cases. Existing Pinball, Cylinder and Cavity trials
  are development evidence for this selection, not a complete baseline suite.
- The stronger Kolmogorov result uses GPPO. PPO near the plotted PPO reward,
  especially a zero-action policy, does not reproduce GPPO. The current Eq.34
  alternative is not an exact Eq.32 GPPO implementation.
- Missing reference information for other families is tracked explicitly.

Paper specialist-policy reproduction is separate from the final AutoResearch
task's one-jointly-trained-checkpoint requirement. Do not package specialist
policies behind a dispatcher and call that a shared model. A shared baseline
must itself be jointly trained and evaluated separately.

## 3. Complete independent replay and per-case acceptance

Train only from the fixed allowed resources. Export code, weights and fixed
preprocessing into a portable package; independently replay frozen snapshots in
the reviewed CFD environment. Preserve raw observations/actions, rewards,
physical metrics, actual step counts, package identity and costs.

The per-case evaluation statistic remains the mean undiscounted native episode
return under the declared initialization protocol. Also report the physical
objective with its declared spatial and temporal window. Do not compare this
return directly with a differently normalized training curve, or substitute
drag reduction for a reward that additionally penalizes lift.

Before acceptance, specify justified tolerances from the matched statistic and
available uncertainty. A hand-picked checkpoint, seed or averaging window is
not a replacement for the declared selection/evaluation rule. Do not choose a
tolerance after seeing the achieved gap. Report best-development and final
checkpoints separately where relevant.

The completion table must contain **every final case**, with: training recipe
and progress, checkpoint, paper target/source, independent test result, metric
parity, gap/uncertainty and acceptance status. No subset average can hide a
missing case or establish full-suite success. A resource/solver failure is not
a fabricated zero model score.

The task-derived 0/1-normalized aggregate is supplementary, not an official
published HydroGym score or proof of paper reproduction. Populate its reference
anchors only after same-protocol measurement and the applicable scientific
review, never by copying plot values with unknown units.

## Public-source recheck, 2026-09-06

- The Hub revision remains `cf5ef6748443ae72e9647f398487e98d421a7243`.
  Its metadata lists 1,471 files, including the two repository-level files;
  the environment payload inventory contains 1,469 files. Filename inspection
  found no CSV/NPZ/policy-weight or named reward/score result files. This does
  not prove no useful numerical information exists inside other assets.
- No 3D Cavity or stenotic-pipe payload directory was found in that snapshot.
  Three DRA2303 directories contain only zero-byte solver markers. An existing
  2D Cavity or a marker is not a substitute for a missing physical environment.
- The directly fetched Nature article HTML exposes two supplementary PDF links
  (supplement and peer review), but no numerical source-data archive link was
  found in its inspected links, despite the data-availability statement.
- Zenodo record `21062222` currently lists the single 65,176,654-byte
  `hydrogym-1.0.1.zip`, MD5 `71a1388cdab3f8b7e1e2b9a7d40f7d63`.
  This is the already inspected code archive, not a newly recovered table of
  all learned-policy test scores.

These are scoped public-source findings, not claims that unpublished results
do not exist. Resources and target scores must both be resolved before the
requested full-suite paper-performance acceptance can be asserted.
