# Verifier and tests

`test.sh` is the Harbor verification entry: it replays `/workspace/submission` across all **7 scenarios × 15 blind perturbation seeds = 105 scheduled episodes** using `sealed/perturbations.json` (`sealed-v0.4`: FAIR-MAST-grounded ranges plus the structural plant mechanisms documented in `instruction.md`) and copies `reward.json` (`score` = `reward` = `successful_episodes / 105`) and `report.json` to `/logs/verifier/`.

Build and run the verifier image from the task directory:

```bash
docker build --target verifier -t tokamak-verifier -f environment/Dockerfile .
docker run --rm --network none --cpus 32 \
  -v /abs/submission:/workspace/submission:ro -v /abs/logs:/logs/verifier \
  --entrypoint bash tokamak-verifier /tests/test.sh
```

`tests/` (including `sealed/perturbations.json` and this `README.md`) is copied into the `verifier` stage only; the `development` image target in `environment/Dockerfile` and `environment/docker-compose.yaml` never copies or mounts `tests/`.

To regenerate `tests/sealed/perturbations.json` with a fresh private key:

```bash
SSI_TOKAMAK_SEALED_KEY="<private-hex-key>" python tests/generate_sealed_perturbations.py \
  --output tests/sealed/perturbations.json
```

The retired `1001..1015` draw is retained solely inside `tests/fixtures/legacy_perturbations_1001_1015.json` for offline regression testing and is never present in any development-visible file.

## CPU-only unit tests (no simulator / no GPU)

```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p 'test_*.py' -v
```

- `test_metrics.py`: shape/X-point metric definitions, windows, success tolerances, steady-state window, `tolerance_time_fraction`, verifier aggregation (every scheduled episode in the denominator, latency gate, invalid package), bounded rerun policy, and step fallback ladder.
- `test_protocol.py`: `tokamak-policy/v1` protocol failure handling with synthetic policies (out-of-bounds, wrong size, NaN, crash, bad observation).
- `test_solver_guards.py`: FreeGSNKE numerical residual work-budget and controllability-redraw guards (skipped automatically when FreeGSNKE is not installed).
- `test_v4_sealed_security.py`: V4 sealed-perturbation expansion (`7 × 15 = 105` unique `(case, seed)` episodes, zero overlap with `1001..1015`), full-tree leak audit outside `tests/`, spy-controller `reset_context` & stderr-fd seed isolation check, and `ssi_index_v3.tokamak()` parity (`< 1e-12`) across normal, partial, failed, void, and boundary fixtures.

## Simulator smoke (minutes, CPU)

```bash
docker run --rm --network none --cpus 4 tokamak-verifier \
  python /opt/tokamak-task/evaluation/smoke_env.py --cases mastu_diverted_hold --steps 10
```

Fallback ladder end to end (FreeGSNKE, two cases × two seeds): `TOKAGYM_TEST_FAULTS` makes chosen steps fail, e.g. `mastu_diverted_hold:0:30:exception:0` (recovered at level 1, bitwise identical trajectory) or `...:timeout:1` (recovered through the linearised step). It is a testing hook and is never set by `test.sh`.

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

The primary scalar score (`score` and `reward` in `/logs/verifier/reward.json`, matching `harness/ssi_index_v3.py:tokamak`) is the **joint operational episode success rate across all 105 scheduled episodes** (`7` equilibrium/tracking scenarios × `15` blind FAIR-MAST perturbation seeds):

$$\text{score} = \frac{N_{\text{success}}}{105}, \qquad N_{\text{scheduled}} = 7 \times 15 = 105.$$

An episode counts as a success if and only if:
1. **Full-horizon survival without machine-protection trip**: the plasma survives the full scenario horizon (`700` steps = `0.35 s`, or `1,200` steps = `0.60 s` for `mastu_sequence`) without vertical displacement / boundary loss (`plasma_lost`), active-coil overcurrent (`coil_limit`), or unrecovered Grad-Shafranov failure (`solver_failure`);
2. **Joint MAST-U / TCV steady-state operational tolerances** after the `100 ms` settling window:
   - 32-point $\psi_N = 0.995$ boundary RMS error $\le 1.0\text{ cm}$,
   - inner separatrix gap error $\le 1.0\text{ cm}$, outer separatrix gap error $\le 1.0\text{ cm}$, and lower X-point height error $\le 1.0\text{ cm}$,
   - relative plasma-current tracking error $|I_p - I_{p,\text{target}}| / I_{p,\text{target}} \le 3\%$;
3. **Real-time PCS inference constraint**: isolated post-replay median policy computation time $\le 0.5\text{ ms}$ (one $2\text{ kHz}$ control period), otherwise the submission is invalid (`score = 0.0`).

**Upstream Literature Provenance**:
- **Closed-Loop Tokamak RL Control & Operational Termination**: *Degrave et al., "Magnetic control of tokamak plasmas through deep reinforcement learning," Nature 602, 414–419 (2022), doi:10.1038/s41586-021-04301-9*, and *Tracey et al., "Towards practical reinforcement learning for tokamak magnetic control," arXiv:2307.11546 (2023)* — establishes raw magnetic-probe/flux-loop observations to PF-coil voltage commands, coil-current machine-protection termination, $3\%$ plasma-current tolerance, hidden per-episode plasma/supply randomization, and sub-control-period real-time actor execution.
- **MAST-U Shape, Gap & Settling Requirements**: *Lvovskiy et al., "Framework for assessment of magnetic equilibrium controller performance on the MAST Upgrade spherical tokamak," Plasma Phys. Control. Fusion 67 (2025), doi:10.1088/1361-6587/ade3fd*, and *Anand et al., "Real-time plasma equilibrium reconstruction and shape control for the MAST Upgrade tokamak," Nucl. Fusion 64, 086051 (2024)* — defines the $\le 1\text{ cm}$ steady-state boundary/gap accuracy and $100\text{ ms}$ settling window on MAST-U.
- **Free-Boundary Evolutive Grad-Shafranov Simulator & Perturbation Calibration**: *Amorisco et al., "FreeGSNKE: A Python-based dynamic free-boundary toroidal plasma equilibrium solver," Physics of Plasmas 31, 042517 (2024)* (FreeGSNKE v3.1.0), calibrated against *FAIR-MAST* (*UKAEA open MAST archive, CC BY-SA 4.0*) and MAST-U solenoid current limits (*Fishpool et al., J. Nucl. Mater. 438, 2013*).

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

1. **Why every scheduled episode—including `solver_failure` and `void` episodes—remains in the fixed `105` denominator (no void exclusion)**:
   In earlier prototype versions (`v2`/`v3`), excluding solver stalls ("void episodes") from the denominator allowed two severe distortions: (a) controllers that drove the plasma toward marginally ill-conditioned equilibria could stall Newton-Krylov iterations before crossing the $20\text{ cm}$ loss threshold and artificially shrink their denominator, and (b) host CPU load affected wall-clock timeouts. In V4 (`tokamak-shape-metrics/v4`), wall-clock step cutoffs are replaced by deterministic residual-evaluation work budgets (`20,000` calls) plus a bounded 2-level exact-replay fallback ladder (`150` NK iterations, then a single linearised step). Any remaining failure or voided episode reflects an unrecoverable plasma state induced by the closed-loop trajectory and therefore counts in the fixed `105` denominator, guaranteeing an invariant denominator across all submissions (`ssi_index_v3.tokamak`).
2. **Why `reset_context` masks the physical perturbation seed instead of removing `seed` from the wire protocol**:
   Keeping `context.seed` in `tokamak-policy/v1` preserves backward compatibility with stateful or stochastic controllers that initialize deterministic internal filters, while replacing the simulator's secret perturbation seed (`env.reset(seed=secret_seed)`) with an uncorrelated public episode index seed (`100000 + episode_index`)—and sanitizing process stderr filenames—eliminates any side channel through which a controller could identify or look up a specific plant draw.
3. **Why smooth native shaping rewards or action-smoothness penalties are not blended into the benchmark score**:
   Following *Degrave et al. (Nature 2022)* and *Lvovskiy et al. (PPCF 2025)*, tokamak discharge qualification is governed by hard physical survival and operational steady-state tolerances rather than arbitrary weighted sums of coil voltage slew or transient overshoot; power-supply slew rate and voltage saturation are already enforced directly in the physical plant model.

### 3. Diagnostic Signals Retained in report.json

While `score` (`successful_episodes / 105`) is the sole primary metric, `/logs/verifier/report.json` (and `results.json` / `reward.json`) retains rich physical and numerical diagnostics for analysis without exposing private perturbation seeds:
- **Continuous proximity companion**: `tolerance_time_fraction` (global and per-case mean share of the post-settling window simultaneously satisfying all five shape/gap/current tolerances);
- **Survival & failure mode breakdown**: `feasible_fraction` (full-horizon survival rate), per-case `outcomes` (`success`, `kept`, `plasma_lost`, `coil_limit`, `solver_failure`), `void_episodes`, `solver_failure_episodes`, `fallback_steps`, and `episode_reruns`;
- **Steady-state physical tracking errors** (averaged over full-horizon feasible episodes per scenario): `shape_rms_cm`, `rin_error_cm`, `rout_error_cm`, `zx_error_cm`, `xpoint_error_cm`, and `ip_error_pct`;
- **Real-time compute telemetry**: isolated post-replay `policy_compute_ms` (and per-case medians) alongside in-replay `policy_compute_ms_under_load`.
