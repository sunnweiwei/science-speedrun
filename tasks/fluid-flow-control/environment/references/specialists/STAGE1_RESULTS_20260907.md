# Five-case reproduction: first measured comparison

Snapshot: **2026-09-07 02:01 UTC**. Manual specialist training, no AutoResearch
agent. None of the five cases is yet accepted as a completed paper reproduction.
The 47-case / 488.77 GB resource release and shared-model task contract are unchanged.

Subsequent reporting follows [the baseline-relative interpretation rules](REPORTING_CRITERIA.md):
include the no-control starting point, achieved improvement and per-case numerical
guidance. This dated snapshot is retained as measured; its visual paper ranges
have not become formal acceptance thresholds.

## Learned-policy results

All five measurements below are real, complete 200-control **fresh-process,
frozen-policy evaluations**, not training loss or no-control validation. The
paper column is a coarse visual range of the corresponding late-time drag
curve, **not a reported time average, uncertainty interval or acceptance band**.
Our statistic is fixed in advance: mean total drag coefficient over controls
101–200. Both columns have the same physical units, but exact paper evaluation
duration and averaging window remain unresolved; a strict numerical gap is
therefore deliberately not calculated.

| Case / method | Published control result: late-time CD, visually approximate | Our last-half mean CD | Our full-episode mean CD | Raw 200-control return | Evaluated training controls | Full training progress |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Pinball 2D Re100 / DDPG | ED Fig.2a red: about 0–0.1 | 0.250396 | 0.404961 | -123.564323 | 80,000, best development checkpoint | Seed42: 100,000/100,000 (100%); seed43 continues, seed44 queued |
| Cylinder 2D Re3900 / PPO | ED Fig.2c blue: about 1.0–1.4 | 1.127054 | 1.148207 | -314.185198 | 20,000, development-selected immutable checkpoint | 37,400/500,000 (7.48%) |
| Cylinder 2D Re1000 / PPO | ED Fig.3a blue, from scratch: about 1.1–1.3 | 1.507044 | 1.528800 | -479.261017 | 400, separate pilot final | New full run: 600/400,000 (0.15%) |
| SquareCylinder 2D Re200 / PPO | ED Fig.3c blue, from scratch: about 1.1–1.2 | 1.500326 | 1.501849 | -317.030812 | 400, separate pilot final | New 600,000-control run initializing |
| Pinball 3D Re150 / PPO | Fig.2a blue: about 0–0.2 | 2.921955 | 2.896420 | -675.423967 | 400, pilot final in a second fresh process | Pilot only: 400/100,000 (0.4%); full run held |

Source: [Nature paper, Fig.2 and Extended Data Figs.2–3](https://www.nature.com/articles/s41586-026-10917-6).
The paper captions identify training curves as three-training-seed means with
one-standard-deviation bands, but test curves as a single policy rollout. Our
best checkpoint from one seed is **not** a reproduction of that three-seed mean.
We do not compare the raw returns above to differently scaled paper training
curves (notably Pinball). All figure readings remain diagnostic estimates.

What this establishes:

- Cylinder Re3900 has learned useful control and reaches the rough plotted drag
  scale. Training and multi-seed reproduction are incomplete; lift, temporal
  coupling and final-figure settings still matter. A lower single-checkpoint
  return magnitude does not establish that we beat the paper.
- Pinball Re100 learned strong drag reduction, but its late drag remains above
  the plotted near-zero curve. The completed DDPG seed42 final checkpoint is
  less good than the selected best: raw return -157.412478, mean CD .430178,
  last-half CD .253360. Both are retained. Corrected TD3 seed42 also completed,
  with best return -218.492449; it is not silently pooled with DDPG.
- Re1000 and Square Re200 values are implementation pilots, not failed full
  convergence attempts. Their long from-scratch PPO runs have now started.
  Square's best pilot checkpoint is at **step 0**; we explicitly report the
  trained step400 final instead of mislabelling an untrained actor as learned.
- Pinball Re150 remains unresolved: the same trained checkpoint gave
  -649.480906 in the training process, -675.476564 in a fresh process and
  -675.423967 in a second fresh process. A two-full-episode frozen-policy
  reinitialization diagnostic is running; no changed reset rule or tolerance
  has been adopted merely to make this pass.

## Separate no-control physics checks

These are **not learned-policy scores**. Paper means are simulation-validation
table entries; exact original averaging durations still need confirmation.

| Case | Paper no-control mean CD | Our 200-control mean CD | Relative difference |
| --- | ---: | ---: | ---: |
| Cylinder 2D Re1000, SI Table 1b | 1.533 | 1.530778 | -0.145% |
| SquareCylinder 2D Re200, SI Table 4a | 1.469 | 1.489440 | +1.391% |

Square lift RMS is .426960 versus .391 (+9.197%): drag agreement alone does
not certify the mesh/flow. Cylinder Re1000 sampled lift half peak-to-peak is
1.448324 versus the paper amplitude 1.449 (-.047%); its RMS is a different
statistic and is not compared to that amplitude.

## Execution and reproducibility

The new Square binding preserves pinned grid/restart and native **independent**
jets, two action inputs at max_control=.03, with 40 public-default u/v probes
(80 values), y-major order. SI 5D gives 225 CFD substeps/control and 200 controls;
omega=.2 follows the cylinder framework at Re200. Hub default substeps=110 and
old author paired-input/.066/87-substep defaults are recorded differences,
not silently interchanged with the new configuration. Final-figure actuator
locations/amplitude and exact sensor metadata remain unverified.

New full runs mount `hydrogym-public-500gb-v1` read-only, pin source snapshots
and embed release/case hashes in their checkpoints. Cylinder Re1000: 400,000
controls / 2,000 episodes, GPU7. Square Re200: 600,000 / 3,000, GPU0. Both use
public-source PPO, paper-derived ReLU/GAE .95 settings, seed42 and bootstrap at
200-control truncation, eval_every4000, final/best fresh-process replays.
The selected baseline image is
`sha256:1eb3775b0b4997b2b9fee6a68a8810f687879978a1fab7f7976d53f62266842d`.

Existing Re3900 and Pinball seed queues continue from their original frozen
sources. Original runs without release-ID metadata are replayed with their own
frozen source; they are not relabelled as new standardized-release experiments.
One attempted old-source replay used an unsupported timing flag and another
used the new release-ID guard on a legacy checkpoint; both were rejected before
rollout. Their outputs are retained; corrected `-r2` replays are distinct runs.

New bounded Pinball investigation on GPU6:
`hg-stage1-pinball2d-ddpg-slow-a050-check-s42-20260907`, 1,200 controls,
`ddpg-slow`, max_control=.5. This tests the larger author-example action range
with slower actor updates after the earlier .5 public-recipe trial became
nonfinite. It is explicit reconstruction tuning, not a recovered paper config;
there is no automatic full-run launch or score assigned before its replay.

Validation this turn: 15 baseline adapter/unit tests, 5 report-evidence tests and
65 task tests passed. Complete reward/action/observation/force trajectories for
Re1000 step400, Square step400 and Re3900 step20000 matched fresh replay exactly
at all 200 steps. These checks validate execution, not paper performance.

## Auditable artifacts

Operator outputs are under `/mnt/workspace/hydrogym/reproduction/`.
`stage1-comparison-20260907-initial.json` contains all five selected evaluations,
checkpoint hashes, protocol values, episode-log hashes and live progress snapshot.
The plan [stage1-runs-20260907.json](stage1-runs-20260907.json) explicitly separates
each evaluated pilot from its new full training run. To take another read-only
snapshot, choose a **new** output filename:

```sh
python baselines/stage1_status.py \
  --runs-root /mnt/workspace/hydrogym/reproduction \
  --plan baselines/stage1-runs-20260907.json \
  --output /mnt/workspace/hydrogym/reproduction/stage1-comparison-NEW.json
```

See [STAGED_BASELINES.md](STAGED_BASELINES.md) for the approved staged plan and
[PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) for the full unresolved-source audit.
No dataset modification, benchmark membership change, Git push, upload,
author contact or agent launch was performed.
