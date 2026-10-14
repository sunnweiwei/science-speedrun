# Tokamak magnetic-control delivery contract

**Interface: `tokamak-policy/v1` (draft, 2026-09-20; environment `tokagym-mastu/v0.3`, 2026-10-05).** The wire format is the same
resident JSON-lines protocol as HydroGym's `hydrogym-policy/v1`; only the protocol
name, the context fields and the physical meaning of observations/actions differ.

Publish one current replay package at `/workspace/submission`. It may be read at any moment, so replace
it atomically (the starter's `publish.py` stages the package next to it and swaps it in with one rename):

```text
submission/
  infer.sh       # Required; starts a resident JSON-lines policy process
  README.md      # Architecture, parameter count, training provenance, results
  ...            # One logical checkpoint, code, config and inference resources
```

## One shared controller

The same learned parameters must serve every case of the suite. Task-conditioned
shared components are allowed; bundling independent per-case controllers behind a
dispatcher is not. Include all learned preprocessing/normalization state. During a
scored episode the parameters and normalization are fixed; episode-local recurrent
state, integrators and other non-learning computation are allowed and must reset
on `reset`. Protocol conformance cannot prove the shared-model rule; it is a
contract obligation checked by code and provenance review.

## Portability

`bash /absolute/path/to/submission/infer.sh` must run from any working directory,
resolve resources relative to the package, and need no network, secret, training
process, workspace path or external checkpoint. The declared verifier image
provides Python 3.12, NumPy, FreeGSNKE (not needed for inference) and
`onnxruntime`; **PyTorch is not installed in the verifier**. Bundle any other
dependency inside the package or export the actor to ONNX/NumPy. Relative symlinks
inside the package are allowed; escaping symlinks are rejected. A snapshot of the
package is copied and hashed before replay.

## Resident protocol

UTF-8, one JSON object per line on stdin/stdout, logs to stderr, flush every reply,
no NaN/Infinity, exactly one outstanding request, echo `request_id`. Lines are
limited to 64 MiB. The process serves many episodes and many cases.

1. `reset` starts an episode:

   ```json
   {"protocol":"tokamak-policy/v1","op":"reset","request_id":"r-1",
    "context":{"environment":"mastu_diverted_hold","family":"hold","seed":3,
               "machine":"MAST-U-like","env_version":"tokagym-mastu/v0.2",
               "control_dt":0.0005,"horizon":700},
    "observation_shape":[585],"action_shape":[12],
    "action_low":[-2000.0,-750.0,...],"action_high":[2000.0,750.0,...]}
   ```

   Reply `{"request_id":"r-1","ready":true}`. `context.environment` names the
   control task; it is not a lookup key for memorized action sequences, and future
   suites may add names.
2. `act` sends one flat observation and expects one flat action of 12 coil
   voltages **in volts**, inside the supplied bounds. The evaluator rejects
   out-of-bounds or non-finite actions as a policy error; it never clips.

   ```json
   {"protocol":"tokamak-policy/v1","op":"act","request_id":"r-2","observation":[...]}
   {"request_id":"r-2","action":[12.3,-40.0,...]}
   ```
3. `close` ends the process: reply `{"request_id":"...","closed":true}` and exit 0.

## Observation layout (`tokagym-mastu/v0.2`)

Concatenated float64 vector, length 585:

| Slice | Content | Units |
| --- | --- | --- |
| 0:102 | flux-loop poloidal flux (102 loops, machine order) | Wb |
| 102:500 | pickup-coil field (398 coils, machine order) | T |
| 500:512 | active coil currents (Solenoid, PX, D1, D2, D3, Dp, D5, D6, D7, P4, P5, P6) | A |
| 512:513 | plasma current (Rogowski-like) | A |
| 513:577 | 32 target boundary control points, (R, Z) pairs | m |
| 577:581 | up to two target X-points, (R, Z) pairs, zero-padded | m |
| 581:583 | X-point validity flags | 0/1 |
| 583:584 | target plasma current | A |
| 584:585 | normalized time in episode | 0..1 |

Diagnostics carry per-episode offsets and per-step noise drawn by the evaluator.
The plasma boundary, X-points and internal profile parameters are **not** observed;
a policy that needs them must estimate them from the magnetics. In the sealed plant some
magnetic channels may read exactly 0 for a whole episode (dead channels) and flux loops
may drift (integrator drift).

Target control points start at the outboard midplane and run counter-clockwise; target
X-points are ordered lowest first. Targets move with the case's vertical waveform and, in
`mastu_sequence`, blend from the reference shape into a second shape. Horizons are 700
steps (1200 for `mastu_sequence`); `context.horizon` gives the value.

## Action

12 coil voltages [V]; limits Solenoid ±2000 V, all others ±750 V (provisional
organizer values, see `tokagym/machine.py`). A first-order supply delay of 0–1
control steps is part of the randomized plant; the sealed plant adds per-episode voltage
offsets, a slew-rate limit and supplies that stick at 0 V near zero coil current (not the
vertical coil). **Coil-current limits end the episode** (outcome `coil_limit`), as machine
protection would; the solenoid range is MAST-U's ±55 kA, the others are the FPDT values.

## Evaluation

The evaluator builds the case's reference equilibrium, applies the sealed
perturbation draw (profile parameters, plasma resistivity, passive-structure
resistance, initial coil currents, diagnostic offsets/noise, supply delay and the plant
mechanisms named in `instruction.md`), lets the plasma evolve under a feedforward hold for
a few steps (the handover), then runs the closed loop for the case's horizon. The score is
the fraction of episodes that keep the plasma within the MAST-U steady-state tolerances
(1 cm shape and gaps, 3% plasma current); per-case physical metrics are reported alongside
(`evaluation/METRICS.md`). Policy-reported scores are never accepted; native training
rewards are diagnostics only.

## Inference limits

- **Latency:** the policy's own computation per step (the protocol round trip minus that of
  a null policy that answers the same observation next to it, in alternating order, on every step) must have a median of at most one
  control period, 0.5 ms; otherwise the package scores 0.
- **Size:** at most 1,000,000 learned parameters for the actor that runs during scoring
  (Degrave et al.'s deployed actor had 266,280). Declare the count in `README.md`; this is
  checked by review, not by the protocol.

Development replay uses the visible randomization and seeds; the trusted verifier
uses `tests/test.sh` with a sealed perturbation file that is not present in the
development image. Subset replays are development evidence, never a suite score.
