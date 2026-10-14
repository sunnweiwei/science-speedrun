# References

- `NATURE2022.md`: notes on the Degrave et al. (Nature 2022) and Tracey et al. (2023) recipe (MPO,
  asymmetric actor-critic, reward shaping, domain randomization, episode chunking): what the starter
  reproduces and what it omits.

The runnable reference controllers are baselines now: `/workspace/baseline/pid/` (privileged and
deployable PID) and `/workspace/baseline/hold_current/`. The notes below on the deployable PID
package are the task author's.

## Building the deployable PID package

```bash
# demonstrations (privileged PID on visible draws; observations + descriptors)
python /workspace/baseline/starter/collect_demos.py --output /workspace/research/demos --episodes-per-worker 2 --workers 24 --solver nonlinear
# package: estimator fit (held-out RMSE printed), virtual circuits, limits, gains, set-point offsets
python /workspace/baseline/pid/build_pid_package.py --demos /workspace/research/demos.npz --output /workspace/research/pid-package
# replay it like any submission
python /workspace/evaluation/harbor_verify.py --submission /workspace/research/pid-package --output /workspace/research/pid-package-check --workers 6
```

Two things decide the gap between the privileged and the deployable controller:

- **Virtual circuits must match the requested shape.** The observation's targets are the
  frozen nominal boundary while every reset re-solves the plasma with perturbed profiles,
  so the controller has to move the plasma 1-3 cm to the targets. With the base-shape
  virtual circuits applied to the P4-scaled shape the first package (v3) lost the plasma
  on every P4 episode within 100 steps; storing the circuits per reference shape and
  ramping the set-points from the first estimate fixed this (v4).
- **Estimator quality.** On the reference demonstrations the held-out RMSE is about
  0.5-0.8 cm per descriptor; the X-point columns must be fitted only on samples whose
  labelled X-point is near the reference (the label jumps to a far-away saddle when the
  divertor configuration is lost), and demonstrations should come from the nonlinear
  plant on visible draws. The set-point offsets in `machine.json` reconcile the
  observation's psi_N = 0.995 contour with the PID's separatrix descriptors.

The coil resistances in `hold_current/policy.py` are read from the vendored machine
description; verify them with `tokamak.coil_resist[:12]` when the description changes.
