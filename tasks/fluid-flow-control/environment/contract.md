# HydroGym controller delivery contract

**Interface frozen: `hydrogym-policy/v1`, 2026-09-06.** The package layout,
request/reply protocol, shared-model rule and frozen-weight replay semantics
below are the v1 delivery interface. Breaking interface changes require a new
protocol version. The environment suite is versioned separately: freezing this
interface does not assert that the full case list or every CFD adapter is ready.

Publish one current replay package at `/workspace/submission`:

```text
submission/
  infer.sh       # Required; starts a resident JSON-lines policy process
  README.md      # Architecture, parameter count, provenance, dependencies, results
  ...            # One logical checkpoint, code, config and inference resources
```

The same learned parameters must serve all cases. A shared policy, a jointly
trained world model and planner, or a task-conditioned shared architecture is
allowed. Packing separate specialist models into one file is not compliance.
The checkpoint format and architecture are otherwise unrestricted. Include all
learned preprocessing and normalization state; do not update them during replay.
Recurrent state and non-learning planning computation are allowed and must reset
between episodes. Code/checkpoint review is needed to verify the one-model rule;
a protocol test cannot prove shared training or parameter provenance.

## Portability and publication

Run `bash /absolute/path/to/submission/infer.sh` from any working directory.
Resolve owned resources relative to the package. No training process, optimizer,
workspace path, network, secret, or external checkpoint may be required. The
declared Docker runtime supplies software dependencies; bundle additional offline
dependencies inside the package if necessary. No setup-time downloads are part
of this v1 contract. The evaluator owns the CFD solver and environment data;
the policy does not launch its own CFD instance during scoring.

Publish complete code/weight combinations, not partially rewritten checkpoints.
Keep old weights, training buffers and optimizer state outside `submission`.
Within-package relative symlinks are allowed; escaping symlinks and special files
are rejected. Only one complete version should remain after publication. A
stable package snapshot is copied and hashed before independent replay, and
checked for mutation afterward. This is not a sandbox against malicious code;
the runner must enforce container/network isolation. Independent replay is not
equivalent to a private/hidden test.

## Resident JSON-lines protocol: `hydrogym-policy/v1`

UTF-8, one JSON object per line on stdin/stdout. Send logs to stderr. Flush every
reply. No NaN or Infinity. There is exactly one outstanding request; echo its
`request_id`. The process persists across an episode and may serve multiple
episodes. IDs and ordering are arbitrary and cannot be used as answer lookups.
Requests/responses are limited to 64 MiB per line. The evaluation release sets
the policy response timeout independently of the research budget.

1. `reset` describes the environment and begins a new episode:

   ```json
   {"protocol":"hydrogym-policy/v1","op":"reset","request_id":"episode-x","context":{"environment":"Cylinder_2D_Re200","family":"cylinder","seed":7},"observation_shape":[2],"action_shape":[1],"action_low":[-1.0],"action_high":[1.0]}
   ```

   Reply: `{"request_id":"episode-x","ready":true}`. Shapes above are a
   protocol illustration, not the benchmark's sensor configuration.
2. `act` gives one current observation as a row-major flat array:

   ```json
   {"protocol":"hydrogym-policy/v1","op":"act","request_id":"step-y","observation":[0.1,-0.2]}
   ```

   Reply: `{"request_id":"step-y","action":[0.01]}`. Action is a row-major
   flat array in **native environment action units**, exactly the product of
   `action_shape`, within the supplied finite bounds. The evaluator does not
   silently clip invalid actions. Observations use the frozen native observation
   pipeline; any additional candidate normalization is packaged with the model.
   These units can be normalized: MAIA's cylinder accepts `[-1, 1]` and applies
   its physical control scale internally. Do not apply that scale twice. Some
   other MAIA environments accept `[0, 1]`; always use the supplied bounds.
3. `close` ends the process. Reply `{"request_id":"...","closed":true}`
   and exit zero. This protocol operation is not a research stopping condition.

`context.environment` and `family` describe the control task, not a per-episode
checkpoint selection instruction. There is no closed ID list in this protocol.
Conforming future inputs may have different values, shapes and IDs; report real
capacity limits in the package. Never substitute memorized action sequences for
closed-loop observation-conditioned control.

The evaluator supplies the observations, applies validated actions, advances CFD
and computes the physical metrics in `evaluation/METRICS.md`, retaining native
rewards only as diagnostics. Policy-reported scores are not accepted. Each
case's horizon, initialization and observation/actuation definition belong to
the frozen evaluation release, not to the candidate.

## External validation

For a direct per-case check, the executable entry point is:

```bash
bash /opt/hydrogym-task/evaluation/evaluate_native.sh \
  --data /data --submission /workspace/submission \
  --case Cylinder_2D_Re200 --output /workspace/research/evaluation-001
```

Run inside the declared Docker runtime. Repeat `--case` or use `--case all` for
the same package across cases. No model-specific evaluator integration is
required beyond this contract. The evaluator snapshots code and checkpoint,
resets the resident policy, sends observations, applies returned native actions,
and reports the fixed per-case `primary_metric`, with native episode returns
retained as diagnostics. It never trains or resumes the model.
The direct replay and Harbor verifier use the same approved resource/runtime
profiles. Neither requires reference anchors or an additional suite.json.
Harbor's tests/test.sh invokes harbor_verify.py for the 26-case verifier suite
(`case_spec.VERIFIER_CASES`: every 2D case and one 3D case per family) and writes
26 named physical values to /logs/verifier/reward.json; cases run side by side, each
on its own GPUs. `--case all` replays the complete 39-case suite. Units and directions
remain per-case; there is no mixed-unit total. An explicit --case subset is
for development only, never a full-benchmark score. See the evaluator README.

The organizer's evaluator first checks the entire declared release, then copies
one stable submission snapshot and uses that same snapshot for **every** case.
The agent supplies actions only; it cannot supply its own benchmark scores.
Every declared episode is replayed using its fixed environment initialization,
observation/reward pipeline, action bounds, control interval and horizon.
Parameters and learned normalization statistics remain fixed throughout replay;
episode-local recurrent state is reset by `reset`.

Validation reports package identity, suite/metric identity, per-case outcomes and
physical primary metrics. A failed policy case remains in the result table. A
missing resource or unimplemented backend is an evaluator/infrastructure gap,
not permission to silently skip that case. Paper-baseline reproduction is a
separate scientific comparison, not a prerequisite for measuring physical metrics.
The automatic protocol checks cannot prove absence of in-memory learning or
independent specialist weights; those remain explicit contract obligations
subject to code and provenance review.

### Fixed reporting: `hydrogym-physical-metrics/v1`

`evaluation/metrics.py` assigns exactly one primary metric to every frozen case;
`evaluation/METRICS.md` fixes units, direction, targets and averaging windows.
CD is primary for the 19 drag-control cases, not for all 39 cases. Other tasks
use L/D, gust load deviation, thrust-angle error, TKE or wall shear. Compare
uncontrolled, model and reference using the same row metric. Training rewards,
losses and terminal observables cannot replace missing physical scores. Do not
mix these quantities into the legacy normalized-return aggregate. The shared
model delivery interface is unchanged.

For `hydrogym-public-research-v2`, the release's `release.json` (`/workspace/data/release/release.json` in development) fixes training/test
membership and source resources. The organizer's executable `case_spec.py`
binds its identity/hash and covers exactly the same 39 cases. Those 39 resource
variants are our public-resource scope, not the paper's original environment
count. Resource selection is frozen; full native replay acceptance is separate.
The source assets total 453,756,315,982 bytes under a 500,000,000,000-byte budget,
shared between training and evaluation. This is not a training-transition,
checkpoint-size or runtime-disk limit. The one-checkpoint and fixed-parameter
evaluation requirements above are unchanged.
Cost tiers do not make cases optional: the shared model must serve all 39. The
verifier scores the 26-case suite, side by side on eight H100s (2 h 21 min, set by
Cube_3D_Re3700; longer when other jobs share the GPUs); the 13 repeated 3D
instances it leaves out cost 2.0-5.2 H100-hours each. Cube_3D_Re3700 keeps a
two-GPU solver binding in the scored suite.

The current implementation and measured acceptance boundaries are documented
in `/workspace/evaluation/README.md`. Conformance to this interface alone is
not a claim that the complete benchmark has already been executed successfully.

Harbor result keys are `CASE/metric_id` and values are the actual physical
quantity; observe each metric's direction rather than assuming all are maximized.
A failed policy/invalid operating regime is retained as an explicit case outcome
in results.json, not a fake physical zero or a dropped case. A case fails when the policy
breaks the protocol, when the simulation turns nonfinite (observation, force or reward) after
the policy has acted, or when the operating regime is invalid. reward.json then flags
`CASE/failed: 1` for it, with no physical metric invented, counts it in `failed_cases`, and keeps
every other case's value. A failed case is the worst outcome on that case. If no case has a
valid value, reward.json holds only `invalid_submission: 1`. Either way the verifier exits 0.
Infrastructure faults (invalid state before the first action, a broken solver, missing cases)
remain separately labeled and exit nonzero.
