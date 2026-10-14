# Deployable full-shape virtual-circuit controller (classical reference, v0.2)

A classical magnetic controller that runs from the observation only (`tokamak-policy/v1`, NumPy,
about 0.1 ms per step). It is the deployable classical reference for `tokagym-mastu/v0.2`; the
learned controllers of the task should beat it. No RL, no neural network.

## Control law (`policy.py`)

1. **Real-time reconstruction.** A linear estimator maps the 513 real-time signals (102 flux loops,
   398 pickups, 12 coil currents, Ip) to the radius of the psi_N = 0.995 boundary along 64 rays from
   (R, Z) = (0.95, 0) m, the lower and upper X-points and the current-centroid height. One set of
   normal equations per reference shape (3 shapes), chosen per step from the observation's target
   points (gain scheduling, as for the virtual circuits). It is fitted on noise-free probe signals
   with the sensor-error covariance (noise, offsets, flux-loop drift) added to the Gram matrix, which
   equals training on infinitely many noisy copies. Channels that read exactly 0 at the first step
   (dead channels) are dropped by re-solving the stored normal equations once per episode.
   Held-out error on noisy, offset, dead-channel observations of the development plant: boundary
   0.44 cm RMS, X-points 0.3-0.4 cm, current centroid 0.26 cm.
2. **Shape.** The errors between the observed 32 target points and the estimated boundary on the
   same rays, plus the two X-point errors (weight 4 points each, ignored if more than 10 cm off),
   are turned into coil-current rates by a Tikhonov-regularised weighted least-squares virtual
   circuit, `dI/dt = 300/s * (J^T W J + lam)^-1 J^T W e`, over the 10 shaping coils and P6. J is the
   static Jacobian of the boundary rays, X-points, gaps and Zcur with respect to the coil currents at
   each reference equilibrium (central finite differences, `jacobians.py`). The shaping coils follow
   their part of the correction; the P6 part, converted to a height with dZcur/dI_P6, moves the
   vertical set-point (outer loop, 150/s), because only P6 can correct an up-down asymmetric error.
3. **Plasma current.** PI on Ip through the solenoid (FreeGSNKE PCS gains).
4. **Coil currents to voltages.** The integrated current references are clipped to 94% of each coil's
   range and to 400 A from the measured current (anti-windup), then
   `V = R I + L (dI_ref/dt + (I_ref - I) / 15 ms)` with the full mutual-inductance matrix.
5. **Vertical (P6).** PID on Zcur * Ip with kz_p = 0.003, kz_d = 5e-6, kz_i = 0.05, toward the
   set-point of step 2 (initialised from the targets' vertical offset). The FreeGSNKE PCS
   gains (0.025, 5e-7) oscillate and lose the plasma once the supply delays the command by one or two
   steps (the v0.2 plant does); `vertical_design.py` shows the chosen gains keep every linearised
   development plant (growth times 2-150 ms) stable for delays 0, 1 and 2.

## Provenance

| step | script | cost |
| --- | --- | --- |
| reference Jacobians (3 shapes) | `jacobians.py` | ~3 min, 1 CPU |
| vertical gain design (40 linearised plants) | `vertical_design.py extract/design` | ~10 min |
| estimator data: 236 episodes of the privileged version of the same controller (simulator boundary instead of the estimator) with random low-frequency target excitation (1.5-2 cm), dev randomization, seeds 6001-6020 and 7001-7016 | `rollouts.py --mode truth --record --params '{"excite": 0.02}'` | ~1.5 h on 112 CPUs |
| estimator fit (92k frames) | `estimator.py fit` | ~2 min |
| package | `build_package.py --params '{"w_x": 4, "k_zt": 150}'` | seconds |

`dev_randomization.json` is the development plant: the visible ranges widened and every v0.2 plant
mechanism switched on at organizer-guessed values (the categories in `instruction.md`, not the sealed
values). Gains were chosen on development seeds 5001-5007 only; the sealed seeds were run once, on the
final package. The data collection uses simulator labels (boundary, X-points, Zcur) as the deployable
PID package used privileged-PID demonstrations; the shipped controller reads only the observation.

`package/` is the built submission (`infer.sh`, `policy.py`, `params.json`, `model.npz`: Jacobians,
reference targets and machine arrays; `estimator.npz`: normal equations). Learned parameters: the
estimator's normal equations, 3 x (513 x 513 + 513 x 69) = 0.90 M stored numbers, of which the
per-episode solved weights are 3 x 513 x 69 = 106 k.

Results: `RESULTS.md`.
