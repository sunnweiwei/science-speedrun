# Verification and regression tests

## Harbor entrypoint

`tests/test.sh` is bundled into the separate verifier image. It invokes the
trusted `harbor_verify.py` on the **26-case verifier suite** (every 2D case, one 3D case per family;
`case_spec.VERIFIER_CASES`) using the artifact at
`/workspace/submission` and read-only data at `/data`. It publishes
`/logs/verifier/reward.json` only when all requested physical metrics are valid.
The detailed replay includes explicit policy/infrastructure errors; no failed
case is silently omitted or filled with a favorable zero.

The native Compose definitions use source/profile preflight, not the retired
normalized-return suite gate. Set `TASK_DATA_DIR` before building/launching.
See [evaluation](../environment/evaluation/README.md) and [contract](../environment/contract.md).

## Fast CPU tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s hydrogym/tests -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s fluid-flow-control/environment/references/specialists -p 'test_*.py' -q
python3 fluid-flow-control/environment/data/public_release.py --check
```

The full pinned task runtime supplies PyTorch/SB3, TorchRL and NumPy; a host
without those dependencies may skip ML-specific tests. Report skips honestly.
Tests include synthetic fixtures for protocol, snapshot/relocation, common
shape capacity, shared training/resume, malformed actions, source/hash changes,
physical extraction/windows, MAIA GPU binding and the Harbor metric bridge.
Synthetic numbers are never published as CFD reference scores.

[docs/VALIDATION.md](../docs/VALIDATION.md) records the actual container and native checks
performed for this PR. Real full-episode specialist evidence lives under
`environment/references/specialists/verified`; CPU tests alone do not establish CFD performance.

The older normalized-R unit fixtures remain regression coverage for retained
historical helpers. They do not define the current benchmark metric or require
a production `suite.json`.

## GPU Sandbox & Blind-Seed Calibration Scripts

- `tests/check_dev_mpi_all_cases.py`:
  - `--mode smoke --data /data --output /tmp/mpi-smoke`: runs `reset()` + 5 zero-action steps across all 26 scored verifier cases (4 HydroGym JAX/JAX-Fluids + 22 MAIA), verifying container-safe OpenMPI (`OMPI_MCA_btl=self,vader,tcp`, `OMPI_MCA_btl_vader_single_copy_mechanism=none`) and dual-rank GPU binding.
  - `--mode full-dual-rank --data /data --output /tmp/mpi-dual-rank`: runs full-horizon zero-control rollouts across all 3 dual-rank MAIA cases (`Cube_3D_Re3700`, `Cylinder_3D_Re3900`, `NACA0012Gust_3D_Re50000_AOA12`), logging peak GPU memory (MiB), wall-clock runtime, and zero-control stability.
- `tests/calibrate_blind_zero_control.py`:
  - `--generate-seeds`: generates 3 cryptographic private perturbation seeds (`!= 123`) per case into `tests/sealed/blind_flow_seeds.json`.
  - `--calibrate --data /data --output /tmp/blind-zero-cal`: replays zero-action across the blind seed roster on GPU and populates calibrated `zero_control_anchors`.

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The physical quantities, averaging windows, actuation bounds, and PDE solver configurations directly follow the **HydroGym** benchmark platform (*HydroGym: A reinforcement learning platform for fluid dynamics*, **Nature** 2026, DOI: `10.1038/s41586-026-10917-6`; upstream repository `dynamicslab/hydrogym` pinned at commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`). Each of the 26 scored cases (`case_spec.VERIFIER_CASES`, spanning every 2D case and one 3D case per flow family across MAIA Lattice-Boltzmann, JAX spectral, and JAX-Fluids compressible solvers) is evaluated strictly on its canonical physical objective in `evaluation/metrics.py`:

- Mean drag coefficient $C_D$ over the post-transient second half of the episode (19 bluff-body wake cases);
- Mean lift-to-drag ratio $L/D$ over the second half of the episode within the admissible $C_D \ge 10^{-6}, C_L \ge 0$ regime (`NACA0012` airfoil family);
- Full-episode mean weighted gust-load tracking error $|C_L - C_{L,0}| + \omega |C_D - C_{D,0}|$ (`NACA0012Gust`);
- Full-episode mean thrust-vector angle error in degrees (`Nozzle2D`);
- Second-half mean turbulent kinetic energy (`Kolmogorov_2D`) and mean wall shear stress (`Channel_3D_Retau180`, with the documented `channel-72cube-local-actuator-clock-v1` actuator-window clock fix).

To synthesize heterogeneous physical units without distorting task balance toward multi-Re families (such as 6 `NACA0012Gust` variants vs. 1 `Channel` or 1 `Kolmogorov` variant), SSI V3/V4 (`harness/ssi_index_v3.py:flow`) computes for each case $c$ its clipped signed relative improvement against the paired uncontrolled zero-action baseline $z_c$:
$$g_c = \operatorname{clip}\!\left(\sigma_c \frac{v_c - z_c}{|z_c|},\, -1,\, 1\right), \quad \sigma_c = \begin{cases}-1 & \text{lower is better}\\ +1 & \text{higher is better}\end{cases}$$
with any diverged, non-finite, or protocol-failed simulation assigned $g_c = -1$. Across blind perturbation seeds, $v_c$ is the equal-weight mean over scheduled episodes (any diverged episode failing the case). Cases are stratified into the **10 flow families** defined by `case.split("_")[0]`, averaged within each family, and averaged equally ($1/10$ weight per family) before top-level `[0, 1]` clipping:
$$\text{signed\_score} = \frac{1}{10}\sum_{f=1}^{10} \frac{1}{|C_f|}\sum_{c \in C_f} g_c, \qquad \text{score} = \operatorname{clip}(\text{signed\_score},\, 0,\, 1).$$

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No ad-hoc actuator-energy or action-smoothness penalty added to the primary physical objective**: In *HydroGym (Nature 2026)*, physical actuator authority is already bounded by native `MAX_CONTROL` / `action_low..action_high` limits and internal actuator ramp time constants (`omega` in gust load tracking, ramp substeps in `Channel_3D`). Adding an arbitrary $L_2$ action penalty on top of $C_D$ or TKE would alter the Pareto optimum away from the published fluid-mechanics formulation.
- **No parameter-count or inference-FLOP cap beyond the episode response timeout**: Multi-geometry closed-loop flow control across 1D/2D/3D sensor layouts legitimately benefits from expressive shared sequence or world-model controllers; only online weight updates during evaluation and multi-checkpoint per-case dispatchers are prohibited by contract, while open-loop trajectory memorization of `seed=123` is prevented physically via sealed multi-seed perturbation rosters and masking the secret simulator seed in `policy.reset(...)`.
- **No exclusion of shared-GPU dual-rank execution**: Allowing two MAIA MPI solver ranks to share a single 80 GB H100 (`HYDROGYM_ALLOW_SHARED_GPU=1`) changes only device scheduling, leaving lattice-Boltzmann floating-point updates, MPI domain decomposition, and physical measurements invariant.

### 3. Diagnostic Signals Retained in report.json

Both `grading.json` and `report.json` (plus `results.json` and `reward.json`) retain full uncollapsed scientific diagnostics alongside `score`:

- Unclipped `signed_score` (`raw_mean_improvement`) and unweighted 26-case `flat_case_mean_improvement`;
- All 10 per-family signed means (`by_family`) and all 26 per-case clipped relative gains (`per_case`);
- Native physical measurements (`CASE/metric_id`) in their original physical units (`CD`, `L/D`, degrees, TKE, wall shear) and explicit `failed_cases` list;
- Complete per-episode native RL returns (`return`, `mean_return`), step counts, wall-clock execution time (`wall_time_sec`), force components (`CD`, `CL`, `CS`), and GPU rank binding receipts.
