# Staged specialist baseline construction

Approved direction, 2026-09-06: reproduce cases with located paper control
results first; once a method is convincingly reproduced, adapt that method with
limited task-specific changes to other selected cases. The second stage is an
organizer baseline, not a claim that the paper reported scores for those cases.
No AutoResearch agent is involved in either stage.

Reporting requirement, 2026-09-07: follow [REPORTING_CRITERIA.md](REPORTING_CRITERIA.md).
Every case needs its no-control starting point, paper improvement, achieved
improvement and case-specific interpretation/acceptance range, not bare CD values.
The proposed 90%/95% improvement-recovery guides are not automatic paper acceptance.

Execution update, 2026-09-07: [five-case measured comparison](STAGE1_RESULTS_20260907.md).
Re1000 and Square Re200 short PPO/fresh-replay checks passed; their respective
400k and 600k from-scratch runs are launched. Existing Pinball2D/Re3900 training
continues. Three-dimensional Pinball is still held at a pilot while full-horizon
reinitialization consistency is investigated. No paper-performance acceptance yet.

This supersedes the earlier requirement that every selected resource variant
must have a reproduced published policy score. It does not change the 47-case,
488.77 GB data release, the single-checkpoint AutoResearch contract or the
requirement to eventually evaluate all selected cases. Specialist checkpoints
remain distinct from the later jointly trained shared model.

## Stage 1: paper-result cases

| Case | Paper comparison | Initial method path | Current next step |
| --- | --- | --- | --- |
| Pinball_2D_Re100 | Extended Data Fig.2a | Existing paper-derived DDPG/TD3/PPO trials; select using complete evidence, not one favorable seed | Continue existing corrected runs and independent-seed comparisons; no paper match yet |
| Cylinder_2D_Re3900 | Extended Data Fig.2c | Existing from-scratch PPO | Continue existing 500,000-control reconstruction; do not restart or relabel |
| Cylinder_2D_Re1000 | Extended Data Fig.3a, blue from-scratch line | PPO | New zero-control, 400-control training and independent frozen-policy pilot |
| SquareCylinder_2D_Re200 | Extended Data Fig.3c, blue from-scratch line | PPO | Bind and validate native actuation/observations before starting training |
| Pinball_3D_Re150 | Fig.2a | PPO candidate | Resolve trained-policy replay discrepancy before long training |

Kolmogorov is deferred: its Fig.3b configuration and exact GPPO remain unresolved.
Existing unrelated-to-this-stage diagnostics may finish, but do not qualify as
stage-1 reproduction or automatically trigger full-suite adaptation. Existing
processes and their frozen source directories are not changed by this plan.

Proceed from uncontrolled physics, to short real-CFD optimization and independent
checkpoint replay, then full training budgets and independent training seeds.
The 400-control check is an implementation pilot, not evidence of convergence.
Use seeds 42/43/44 for eventual replication as an explicit organizer choice;
the paper gives the number of seeds, not this full list.

Keep raw training curves, physical objectives and complete evaluation rollouts.
Test returns must retain their native units; scaled learner rewards and visually
estimated figure endpoints are not acceptance targets. Report full-episode and
predeclared last-half drag/lift statistics separately, alongside plotted policy
trajectories where source data permit. The exact paper test windows/normalization
remain unresolved, as listed in [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md).
At the stage-1 review, describe the achieved physical-performance agreement,
seed variability and every remaining setup difference. Do not label a process
exit, successful checkpoint load, or one good point as paper reproduction.

## Cylinder Re1000 pilot recipe

- Data: unchanged `hydrogym-public-500gb-v1`, bound read-only through
  `HYDROGYM_BASELINE_DATA_DIR`. No new fields, mesh coarsening or outside data.
- From-scratch PPO, same existing `public-source` learner factories and paper
  ReLU/GAE .95 corrections; no Re200 pretraining or transfer checkpoint.
- SI 5A: omega=1, 350 CFD substeps/control, 200 controls/episode. Hub YAML has
  225 substeps; this is a recorded override, not overwritten source data.
  The base/fine-grid subcycling/physical-time equivalence is not yet certified.
- Author CylinderBase (1000,2) default sensor coordinates: x=linspace(1,7,13),
  y=linspace(-.75,.75,5), y-major meshgrid order; 65 u/v probes, 130 values.
  Native U_inf scaling is retained, no unrelated cached normalization.
  This is a public-default reconstruction, not recovered final-figure metadata.
- Native paired zero-net-mass-flux jets, one action, max_control=.03. Preserve
  source ramping, grid, restart and coefficient normalization.
- Bootstrap at the 200-control truncation boundary. Pilot seed42, 400 training
  controls, eval_every400, raw unscaled evaluation and fresh final/best replays.
- Suggested eventual plotted-range budget: 400,000 controls = 2,000 episodes.
  It is not launched automatically by the short check; review physics, replay,
  early learning and measured cost first.
- SI Table 1b's CD=1.533, St=.234 and lift oscillation amplitude=1.449 are
  uncontrolled-flow checks. Lift amplitude is not RMS. Do not tune a tolerance
  after inspecting the result or confuse these values with an RL target.

Example from repository root (the launcher refuses existing output names):

```sh
HYDROGYM_BASELINE_DATA_DIR=/mnt/workspace/hydrogym/releases/hydrogym-public-500gb-v1 \
  bash hydrogym/baselines/run.sh RUN_ZERO 7 zero \
    --environment Cylinder_2D_Re1000 --protocol author-paper \
    --bootstrap-time-limits --episodes 1 --seed 42
HYDROGYM_BASELINE_DATA_DIR=/mnt/workspace/hydrogym/releases/hydrogym-public-500gb-v1 \
  bash hydrogym/baselines/run.sh RUN_PILOT 7 train \
    --environment Cylinder_2D_Re1000 --protocol author-paper \
    --learner public-source --algo PPO --recipe public --seed 42 \
    --bootstrap-time-limits --steps 400 --eval-every 400 --evaluate-published
```

Run the two commands sequentially after reviewing the first result. Each run
freezes its learner source; release identity and case-file manifest digest are
embedded in new protocols and checked again during independent policy replay.
Legacy experiment roots remain replayable without retroactive release labels.

## Stage 2: adapted organizer references

Start after the stage-1 method review, not simply after the pilot finishes.
Keep the validated algorithm and learner implementation. For each remaining
case adapt the necessary observation/action dimensions and scaling, temporal
coupling and normalization; limited learning-rate, exploration and training-budget
tuning is allowed, with the tried choices and costs recorded. Do not change the
fixed evaluation reward, restart set or case membership to improve a score.

Every score is labelled `adapted-organizer-baseline`, with its own immutable
checkpoint, inference code, preprocessing, resolved config, training provenance
and independent native replay. Retain unsuccessful trials. If the method does
not transfer, report the gap and review further changes rather than silently
switching to another research program or dropping the case.

Record paper-reproduction evidence and adapted reference scores in separate
columns. Only same-evaluation-protocol measurements can later populate reference
anchors. A shared model baseline, if built later, must actually be jointly trained;
this specialist collection cannot be wrapped as the agent's shared checkpoint.
