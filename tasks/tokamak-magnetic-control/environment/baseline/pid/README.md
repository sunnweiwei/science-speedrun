# Baseline: PID / virtual-circuit controllers

- `pid_reference.py` is the classical PID + virtual-circuit controller built from FreeGSNKE's
  pulse-design tool. It uses **privileged** plasma descriptors computed from the simulator's
  equilibrium, so it is a reference number, not a protocol submission. It also generates the
  starter's demonstrations (`../starter/collect_demos.py`). Since v0.2 its virtual circuits and
  feedback include the lower X-point height, as MAST-U's controller does; on visible seed 0 of
  `mastu_diverted_hold` this took the steady-state shape error from 5.63 to 0.61 cm and the
  X-point height error to 0.27 cm. `--no-zx-feedback` reproduces the v0.1 reference.
- `build_pid_package.py` with `pid_package_policy.py` makes the same control law **deployable**.
  A ridge-regression estimator from the 515 real-time signals replaces the privileged
  descriptors. Virtual circuits are stored per reference shape, and set-points are ramped from the
  observation's targets. The output is a protocol-conforming package, and it is also the starter's
  residual-mode base controller. It still leaves the X-point height out of its virtual circuits
  (`build_pid_package.py`, `pid_package_policy.py`) and was not rebuilt under v0.2; rebuild it
  with Zx controlled, as the privileged reference now is.

```bash
python /workspace/baseline/starter/collect_demos.py --output /workspace/research/demos --episodes-per-worker 2 --workers 24 --solver nonlinear
python /workspace/baseline/pid/build_pid_package.py --demos /workspace/research/demos.npz --output /workspace/research/pid-package
python /workspace/evaluation/harbor_verify.py --submission /workspace/research/pid-package --output /workspace/research/pid-package-check --workers 6
```

`RESULTS.md` has both controllers on the sealed seeds. `/workspace/references/README.md` explains
why the deployable controller trails the privileged one: the virtual circuits must match the
requested shape, and estimator quality limits the vertical loop.
