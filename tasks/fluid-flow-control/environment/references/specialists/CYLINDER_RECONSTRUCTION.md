# Cylinder Re3900: public-source reconstruction protocol

## Subsequent stage-1 extension: Re1000, 2026-09-06

The separate `Cylinder_2D_Re1000` comparison is now runnable; see
[STAGED_BASELINES.md](STAGED_BASELINES.md) for the recipe and staged acceptance.
It uses 65 public-default u/v probes in y-major order, paper 350 CFD substeps,
200-control external time limit, omega=1, and source max_control=.03. It is
not a transferred Re200 policy or a relabelled Re3900 run.

`hg-stage1-cylinder1000-zero-s42-20260906` completed 200 controls on GPU7 using
the read-only standard 47-case release and exited0. Full-episode mean CD was
1.5307775036 versus SI Table1b 1.533 (-0.1450%); half peak-to-peak CL was
1.4483242382 versus reported oscillation amplitude1.449 (-0.0466%). The latter
is sampled amplitude, not RMS. The run's RMS CL was1.0450031889. Episode time
was125.3487seconds; startup/teardown outside that timer are not included.
These are promising uncontrolled-flow checks, not proof of mesh/time-window
equivalence, trained-policy performance or full-suite success. No tolerance was
retroactively chosen to turn these numbers into an acceptance claim.

The trajectory SHA256 is
`155320216b1b9bef8de5e145de8982e0e36a87ec2c795c971ed64200e5e1e5e5`.
`hg-stage1-cylinder1000-ppo-check-s42-20260906` starts from random weights and
is a bounded400-training-control PPO pilot, followed by fresh final/best replay.
The eventual400k-control plot-range reconstruction is not launched by this
pilot. The earlier Re3900 experiment and source below remain unchanged.

## Earlier Re3900 record

This extends manual organizer work to the paper's 2D circular-cylinder comparison,
not the rotary-cylinder task. No successful solver or performance match is claimed
until the actual runs below exist and their metrics have been checked.

- Pinned `Cylinder_2D_Re3900` public MAIA assets, unchanged. Native paired jets
  impose zero net mass flux; one normalized action, physical limit 0.03.
- Supplementary Section 5A: 200 controls per episode, 600 CFD steps per action,
  native reward `-|CD| - |CL|`. Public YAML differs: 500 substeps and 201 controls
  under the pinned implementation's completion check. Overrides are recorded.
- Sensors reconstruct the author's published Re3900 2D class defaults:
  x=linspace(1,8,4), y=linspace(-0.75,0.75,3), u/v, normalized by freestream velocity.
  These are not asserted to be the final plotted experiment's exact sensors.
- The public mesh places the cylinder at the origin in a [-8,24] x [-10,10]
  box. Its max level is 12 and domainLength=32, giving dx=0.0078125. The supplement
  describes a different domain/refinement (up to level 14). Uncontrolled CD=1.68
  in Supplementary Table 1 is a physical check, not proof of mesh equivalence.
- Pinball and cylinder native `get_reward()` return different diagnostic force
  units. The logger converts cylinder's raw lattice forces using the exact same
  density, velocity and projection length as its native reward implementation.
  No reward is replaced, reweighted after replay, or scaled for reported scoring.

First validate uncontrolled physics:

```sh
bash baselines/run.sh hg-cylinder2d-zero-paper-20260906 GPU zero \
  --environment Cylinder_2D_Re3900 --protocol author-paper --episodes 1
```

Then use `train --learner public-source --algo PPO|DDPG|TD3` with an explicit
budget/seed and `--evaluate-published`. Network/optimizer defaults still originate
from the pinned public Pinball YAML plus supplement-derived corrections. That is
a documented fallback, not a discovered original cylinder training configuration.
Figure budget and physical agreement must be reviewed before full training.

Cavity requires a separate review: public Re4140 assets are not the paper's
Re4200 case; the current library uses absolute deviations whereas Supplementary
Eq. 36 specifies squared, reference-standardized deviations. Do not silently use
the public cavity reward and label its result a paper reproduction.

## Native evidence and queued learning

`hg-cylinder2d-zero-paper-20260906` completed200 actual zero-control steps:
return-566.9162190, meanCD1.6235048, RMSCL1.3462135, wall191.34seconds.
That establishes a usable native environment, not trained baseline performance.

`queue-cylinder-public-20260906.sh` waits for the existing GPU5 Pinball run to
finish, then checks a400-control public-source PPO and independent final/best
replays. On successful completion it launches500,000 controls /2,500 episodes
using the checked pilot's frozen source. This explicitly reconstructed budget
covers the plotted extended-data range; the exact original figure configuration
was not recovered. The measured zero episode implies roughly133GPU-hours of
CFD for2,500 episodes on one H100, before learner/evaluation overhead, so this
is a long training job rather than a quick solver test. No cylinder policy has
been trained yet at the time this queue was prepared.
