# Three-dimensional Pinball Re150

This is the Fig. 2a task, separate from Extended Data Fig. 2's 2D Re100 task.
Current evidence does not establish paper-performance reproduction.

**18:21 diagnostic update:** two fixed-action20 episodes in the same GPU process
return -58.32662165 and -58.34666196. Their first next-observations match exactly,
but first rewards differ by .04916487; CPU8 reproduces this first-step behavior.
Fresh-process repetitions and reset-only double-zero-control diagnostics are
running. No root cause or fix is established. Public old author 3D properties
disable control ramping, while the pinned current asset enables it (.95).
Other old/current differences include solver method, domain and restart, so
neither swapping the whole old properties file nor attributing all differences
to ramping is justified. Existing source assets and learners are unchanged.

**17:24 correction:** the trained400 checkpoint failed exact independent replay:
training-process return-649.48090607 versus fresh-process-675.47656366, despite
identical protocol/checkpoint and first transition. First divergence is step2's
next observation (max1.1280e-4 across89 components), before action feedback differs.
The full100k queue and its preceding capacity comparison are now stopped, not
running. Independent initial-best replay continues. Two fresh GPU processes,
each with two20-control fixed-action episodes, and a matching CPU8 diagnostic
are queued to distinguish reset/cross-process/controlled-flow differences.
No cause is established yet. See reproduction/pinball3d-trained-replay-mismatch-20260906.json.
The allocation queue described below is a historical plan, not the active queue.

Use `compare_trajectories.py` to quantify complete replay differences. It requires
the full declared horizon, ordered steps and matching finite field shapes; it
does not declare an acceptable tolerance or paper success. For example:

```sh
python baselines/compare_trajectories.py \
  --left RUN/evaluation-steps.jsonl --left-training-step 400 \
  --right RUN/independent-final/evaluation-steps.jsonl --expected-steps 200
```

Select explicit `--fields reward drag lift` only for a justified force-only
comparison across different probe layouts; do not imply observation parity.

Measured cost update(2026-09-06 16:35 UTC): the corrected pilot collected400
training controls in5042.4040sec, excluding its initial evaluation, or
12.6060sec/control on one H100. Linear extrapolation is14.59days for100000
training controls plus roughly.73day for25 development evaluations of200
controls, before independent final/best replays and other overhead. This is an
extrapolation from two episodes, not a promised completion time. It supersedes
earlier shorter-runtime estimates. The actual trained400 checkpoint is finite
with actorparameterdeltaL2=1.80910894. Its trained400 evaluation returned
-649.48090607; independent final/best replay is underway.

Upstream MAIA examples explicitly support multiple solver ranks
(`-np1 python ... : -npN maia ...`); their Docker script describes NPROC_MAIA
as GPUs or CPUcores. Multi-rank GPU allocation, grid/restart compatibility,
throughput and physical replay parity have not been tested here. No current
job is changed or sped up merely on the basis of that documentation.

At17:16, the waiting full-run queue was replaced by a bounded allocation-only
comparison before100k training. After the current pilot/replays finish, use R9
for two GPU7 zero20 controls at original80M and private20M maxNoCells capacity.
The supplied16,465,722-cell grid and all physical parameters remain unchanged.
The queue adopts20M only if all physical/probe/action trajectories agree within
1e-6 and the measured mean step time (excluding first step) is at least10% lower.
Otherwise retain the original allocation; invalid checks stop the queue before
full launch. This is a short throughput diagnostic, not paper performance or
a promised long-run speedup. See reproduction/queue-pinball3d-capacity-then-full-20260906.sh.

- Use the pinned public `Pinball_3D_Re150` MAIA mesh and restart, unchanged.
- The supplement describes 200 CFD substeps per action and 200 actions per
  episode. The public asset YAML uses 175 substeps and ends after 201 actions
  under the native completion check. The manual `author-paper` protocol records
  the explicit 200/200 choice, omega=1 and the asset's max rotation speed=0.5.
- The author's pinned `torchRL_examples/ppo.py` contains a runnable-coordinate
  3D branch: x=linspace(2,8,7), y=linspace(-.75,.75,5), z=linspace(-1.5,1.5,5).
  Preserve its meshgrid flatten order (y,x,z), giving 175 probes / 350 u,v inputs.
  Their correspondence to the original figure's final sensors is not disclosed.
  No cached 3D normalization vectors were found; use the native freestream
  normalization and record that choice, not the unrelated 2D cached vectors.
- Reuse public author network/loss factories. The Pinball 2D YAML with the
  documented supplement corrections is an explicit learner fallback for 3D,
  not a recovered original 3D configuration. Keep each recipe/seed separate.
- Fig. 2a plots 500 training episodes. A 100,000-control run supplies that count;
  the public 2D YAML's collector total_frames=200,000 is a different budget.
- Prior zero-control diagnostics used a different reconstructed 192-probe set.
  Those remain physical drag/lift checks, not matching observation-protocol
  anchors for the new 175-probe runs.

Run a real 400-control public PPO check, then independently reload final and
best checkpoints in fresh CFD processes. Only after those checks pass, launch
the full100,000-control training run with the audited learner and physics.
The current R9 comparison adds allocation/timing/finite-observation checks;
its public learner and time-limit implementation are byte-identical to the pilot:

```sh
bash baselines/run.sh RUN_NAME GPU train --environment Pinball_3D_Re150 \
  --protocol author-paper --learner public-source --algo PPO --recipe public \
  --steps 400 --eval-every 400 --seed 42 --bootstrap-time-limits --evaluate-published
```

Successful execution is not acceptance of either the physical or learned paper
performance. In particular, a long uncontrolled diagnostic does not provide a
trained baseline or justify rescaling measured drag to a target table value.
