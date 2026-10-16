# Baseline: starter (residual SAC on a deployable base controller, or BC-from-PID + SAC)

A research starting point, not a competitive result (the recorded run and its sealed results: `RESULTS.md`). One learner, two modes:

- **Residual mode** (`--residual-package PKG`, recommended): the workers run a deployable base
  package as the base controller and the actor learns a bounded residual,
  `action = clip(base(obs) + tanh(actor(obs)) * residual_scale * vmax)`. Use
  `baseline/shape_vc/package` (the classical v0.3 reference, 17/35 sealed); any package whose
  `policy.py` defines `Controller` with `reset(low, high)` / `act(obs)` works. The published
  package bundles the base (its `policy.py` as `pid_core.py` plus all its files) with the residual
  actor (`actor.npz`, `config.json`, `predict_residual.py` as `policy.py`).
- **Direct mode** (default): privileged PID demonstrations are behaviour-cloned into the actor,
  then SAC continues with a BC regularizer toward the demonstrations.

| File | Role |
| --- | --- |
| `common.py` | observation scaling, NumPy actor (deployable), save/load of `actor.npz` + `config.json` |
| `workers.py` | simulator worker processes: one environment per case, demonstration episodes first, then actor + Gaussian exploration; episodes to a queue; weight refresh from `actor.npz`; heartbeats |
| `train.py` | learner: replay buffers (demos and all), normalizer fit, BC, squashed-Gaussian SAC with twin critics on `[obs, privileged descriptors, action]`, worker watchdog, snapshots, publication |
| `publish.py` | atomic copy of the actor (+ PID core in residual mode) with `policy.py`, `infer.sh`, README into `/workspace/submission` |
| `predict.py`, `predict_residual.py` | the resident `tokamak-policy/v1` servers shipped as `policy.py` (NumPy only) |
| `collect_demos.py` | privileged-PID demonstrations to disk (for the package estimator / BC) |
| `init_random.py` | random-init package (untrained anchor, plumbing check) |

## Minimal loop

```bash
# 1. plumbing: random package -> replay one case
python /workspace/baseline/starter/init_random.py --actor-dir /workspace/research/r0 --submission /workspace/submission
python /workspace/evaluation/harbor_verify.py --submission /workspace/submission \
    --output /workspace/research/check-r0 --cases mastu_diverted_hold --seeds 0 --workers 1

# 2. residual training on the nonlinear plant, on top of the shape_vc reference
python /workspace/baseline/starter/train.py --run /workspace/research/run-001 --workers 64 --hours 5 --solver nonlinear \
    --residual-package /workspace/baseline/shape_vc/package --residual-scale 0.05 \
    --demo-episodes 1 --demo-min-episodes 32 --critic-warmup 1000 --publish-every 24 --snapshot-every 24

# 3. learning curve of the snapshots on visible draws
python /workspace/evaluation/curve.py --run /workspace/research/run-001/run --output /workspace/research/curve-001 \
    --randomization visible --seeds 0 1 2 --workers 6 --every 2 --watch

# 4. independent replay of the published package on all cases, visible draws
python /workspace/evaluation/harbor_verify.py --submission /workspace/submission \
    --output /workspace/research/full-001 --workers 6
```

`--solver linear` re-linearizes around the reference state (threshold 0.05) and is
1.3-2x cheaper per step, but the PID core keeps the plasma on only a third of the
linear-plant episodes against most of the nonlinear ones, so residuals learned there do
not transfer well; train on the nonlinear plant. `--horizon` shortens development
episodes; published candidates must be trained for the full horizon.

## Design notes

- Actor: MLP 585 -> 256 -> 256 -> 256 -> 12, tanh, scaled to the voltage limits
  (times `residual_scale` in residual mode); ~290k parameters; deterministic NumPy
  inference (~50 us per step). The last layer starts near zero, so the initial policy is
  the base controller (residual mode) or a small random voltage (direct mode).
- Critic input adds six privileged descriptors from the simulator (shape RMS, X-point
  error, Ip error, axis offsets, coil violation): the asymmetric actor-critic of
  Degrave et al.; the actor never sees them.
- Demonstrations: the first `--demo-episodes` episodes of every worker run the base
  controller (residual mode) or the privileged PID (direct mode) without exploration
  noise. Once `--demo-min-episodes` have arrived, the observation normalizer is fitted on
  them, BC runs (direct mode only) and the first package is published.
- Actor objective, after `--critic-warmup` critic-only updates:
  `maximize q_weight * Q(s, pi(s)) / mean|Q| - alpha * log pi - bc_weight * reg`, with
  `reg = mean tanh(residual)^2` (residual mode) or the MSE to the demonstrations (direct
  mode). The `mean|Q|` normalization is the TD3+BC device (Fujimoto & Gu, 2021): without
  it the critic's extrapolation error outside the demonstrated actions swamps the
  regularizer and the policy collapses within a few thousand updates (we observed exactly
  that in earlier runs). `--alpha-lr 0` freezes the entropy coefficient at
  `exp(--init-log-alpha)`.
- Targets are clipped to `[-q_clip, q_clip]`; gradients are clipped at norm 10.
- Exploration noise is Gaussian in the workers (`--explore-noise`, 2% of the voltage
  limit by default) on top of the deterministic actor.
- Publication rule: the demonstrator's own training-window score (mean feasibility,
  then mean last-half shape RMS over the last ten episodes per case) is the bar; every
  `--publish-every` episodes the policy's window is compared with the best so far and
  improvements are published. This uses visible development feedback, as the
  instruction allows; it is not a sealed score. Snapshots (`--snapshot-every`) are
  exported regardless of quality so that `evaluation/curve.py` can draw an honest
  learning curve.
- Workers heartbeat every 25 steps; a worker silent for `--worker-timeout` seconds is
  killed and respawned (FreeGSNKE can hang in a degenerate solve and swallows
  in-process timeouts).
- Known gaps: no recurrent state, no reward thresholds/SmoothMax, no episode chunking,
  no ONNX export, no evaluation on sealed draws (impossible by design).
