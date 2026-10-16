# Independent physical evaluation

The production entrypoint is `harbor_verify.py`; Harbor invokes it through
`tests/test.sh`. Development uses the same code. It executes the candidate's
resident JSONL `infer.sh` in a new process, sends observations, validates native
actions, advances the trusted simulator, and calculates the fixed physical
primary metric. It never loads candidate-provided scores or trains the model.

## What is fixed

- Source membership and bytes: `../data/release.json` (39 cases).
- Solver, observation/action spaces, initialization, native horizon and GPU
  requirement: `case_spec.py`, derived from the approved release.
- Per-case metric, unit, direction and window: [METRICS.md](METRICS.md).
- Delivery: [hydrogym-policy/v1](../contract.md), one package for all cases.

No `suite.json`, paper anchor or private organizer path is required.
`evaluate.py` / `run_maia.py` retain older helper and synthetic-test code;
their legacy normalized-return CLI is **not** this task's grading entrypoint.

## Direct development commands

Inside the declared runtime:

```bash
# No CFD: check all source/profile bindings.
bash /workspace/evaluation/evaluate_native.sh --data /workspace/data/release --case all --preflight

# One explicit development subset; a fresh output directory each time.
bash /workspace/evaluation/native_runtime.sh maia python /workspace/evaluation/harbor_verify.py \
  --data /workspace/data/release --submission /workspace/submission \
  --case Cylinder_2D_Re200 --output /workspace/research/check-001

# Same snapshot across the scored 26-case verifier suite (the default; cases run side by side, one per
# GPU set). Add --case all for the complete 39-case suite.
bash /workspace/evaluation/native_runtime.sh maia python /workspace/evaluation/harbor_verify.py \
  --data /workspace/data/release --submission /workspace/submission \
  --output /workspace/research/full-001
```

The outer runtime can be `maia`; each case subprocess selects MAIA, JAX or
JAX-Fluids itself, clearing incompatible library settings. Raw replay without
Harbor reward export is also available via `evaluate_native.sh --case ...`.

## Clean-container evaluation

Build from the repository root:

```bash
docker build --target verifier -t hydrogym-verifier -f fluid-flow-control/environment/Dockerfile fluid-flow-control
mkdir -p /absolute/evaluation-output
docker run --rm --network none --gpus all --shm-size 32g \
  --mount type=bind,source=/absolute/prepared-data,target=/data,readonly \
  --mount type=bind,source=/absolute/published-model,target=/workspace/submission,readonly \
  --mount type=bind,source=/absolute/evaluation-output,target=/logs/verifier \
  --entrypoint bash hydrogym-verifier /tests/test.sh
```

Replace the absolute paths. The logs directory must be writable and must not
contain a prior reward.json/replay. Only the published model and prepared data
are needed: not the training directory, author checkpoint path or host cache.
The dedicated verifier image bundles its trusted test and evaluation code.
Development-visible evaluator changes do not alter that separate image.

## Outputs and failure semantics

- `reward.json`: flat labeled numeric physical values, e.g.
  `{"Cylinder_2D_Re200/drag_cd": 1.1395}` for an **illustrative subset**.
  Verifier grading has 26 keys (39 with `--case all`). The value is CD, **not** training return.
- `results.json`: same values with units/directions/windows, every case outcome,
  complete episode records, package/data/profile/metric/evaluator identities.
- `grading.json`: `scope=verifier-suite`, `full-39` or `development-subset`, case count and
  `aggregate_score=null`. There is deliberately no mixed-unit total.
- `partial-results.json`, per-case logs/progress: durable diagnostics.
- `grading-error.json` / `error.json`: explicit policy versus infrastructure errors.

Directions differ: L/D is maximized; CD, load/angle error, TKE and wall shear
are minimized. Harbor supports labeled numeric metrics; consumers must use
these documented directions instead of averaging unlike units or assuming
every raw value is a higher-is-better scalar. See the
[Harbor verifier format](https://docs.harborframework.com/core-concepts/tasks/verifier).

Every requested case remains in results. Bad protocol/actions, policy exit,
timeout or invalid L/D operating regime do not become a fake physical score.
Infrastructure failures are distinguished from policy failures. If any requested
case lacks a valid physical metric, the verifier fails explicitly and does not
publish an accepted reward.json. Successful cases remain available diagnostically.

The evaluator validates source/profile bindings, snapshots the entire package,
uses that same snapshot across cases, and checks package/source/code identities
again afterward. It requires complete fixed-length episodes, not an arbitrarily
short prefix. Subset checks never certify the full suite. No silent fallback to R.

## Scope and resource cost

The current specialist evidence has 39/39 paired full-episode independent
results. Those comparison checkpoints are not one
joint model. Current fixed seed/source initializations do not constitute an
independent held-out sample population or establish statistical confidence.
Reference selection used visible development feedback.

Serial replay needs up to two distinct H100s for the largest cases; task.toml
allows eight and a conservative seven-day verifier watchdog. More GPUs alone
do not parallelize the current sequential evaluator. Large MAIA grids need
substantial host RAM; do not infer resource use from policy parameter count.

The single-model and no-online-learning rules are contract obligations requiring
code/provenance review; protocol conformance cannot prove them. This is ordinary
scientific evaluation, not a sandbox against malicious inference code. An agent
harness must enforce the appropriate container/network boundary.
