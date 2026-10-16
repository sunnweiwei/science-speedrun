# Baseline: one shared PPO starting point

This is runnable starter code for **one shared checkpoint**, not a bank of
specialist policies. It pads observations/actions, supplies an observation mask
and task conditioning, and updates the same SB3 PPO actor/critic across training
phases. The architecture is a starting point, not a reproduced SOTA result.

## Train, publish, resume on another case

Inside the development image, with the prepared release at `/workspace/data/release` (also reachable as `/data`):

```bash
# A real initial experiment. Each output directory must be new.
bash /workspace/evaluation/native_runtime.sh maia python /workspace/baseline/starter/train_native.py \
  --case Cylinder_2D_Re200 --steps 400 --seed 42 \
  --resume /workspace/research/shared-ppo.zip --submission /workspace/submission \
  --output /workspace/research/shared-cylinder-001

# Reuse exactly the same checkpoint; do not initialize a per-case model.
bash /workspace/evaluation/native_runtime.sh maia python /workspace/baseline/starter/train_native.py \
  --case Pinball_2D_Re100 --steps 400 --seed 42 \
  --resume /workspace/research/shared-ppo.zip --submission /workspace/submission \
  --output /workspace/research/shared-pinball-002

# JAX and JAX-Fluids use their runtime profiles, with the same shared PPO file.
bash /workspace/evaluation/native_runtime.sh jax python /workspace/baseline/starter/train_native.py \
  --case Kolmogorov_2D --steps 400 --seed 42 \
  --resume /workspace/research/shared-ppo.zip --submission /workspace/submission \
  --output /workspace/research/shared-kolmogorov-003
```

Use runtime `maia`, `jax`, or `jax-fluids` as declared by the case.
`train_native.py` uses all 39 fixed shape capacities and the exact native
observation/action/physics adapter. The finite horizon bootstraps the critic
as a time limit without changing the simulator. Training steps/publication
intervals are positive multiples of the 200-step PPO rollout.

The training command is one finite experiment **inside continuing research**.
Sequential phases can forget earlier tasks; improve allocation, replay,
architecture or optimization using cross-case evidence. A 400-step smoke run
is not a useful convergence budget or a stopping rule for the agent.

## Portable publication and independent evaluation

`train.py` persists resume/optimizer state outside the submission.
`publish.py` atomically updates `policy.pyz` (code, config and one checkpoint),
with a stable `infer.sh`. The inference package needs neither training directories
nor author source. It uses the resident JSONL [contract](../contract.md).

```bash
bash /workspace/evaluation/native_runtime.sh maia python /workspace/evaluation/harbor_verify.py \
  --submission /workspace/submission --case Cylinder_2D_Re200 \
  --output /workspace/research/replay-001
```

Omit `--case` for all 39. Values in `reward.json` are the fixed **physical**
metrics, with directions/units in `results.json`; not training return and not
a mixed-unit sum. A new verifier container must receive only the published
package and prepared data, not the live training workspace.

For a genuinely random (untrained) package to test the protocol:

```bash
bash /workspace/evaluation/native_runtime.sh maia python /workspace/baseline/starter/init_random.py \
  --seed 0 --submission /workspace/research/random-submission
```

This uses standard SB3 initialization, not seed selection or zero actions.
The published specialist table's uncontrolled baseline is **zero physical
control**, which is different from this random-policy integration fixture.

## Verified specialist references

See `/workspace/references/specialists/README.md` for the runnable public-source
PPO/DDPG/TD3 baseline path, and `/workspace/references/results/README.md` for
the 39 completed results. Reuse algorithms, architecture and allowed training
experience, but **do not submit separate per-case models behind a dispatcher**.
No pretrained reference weights are supplied. Historical cavity and old
paper-profile scripts are not part of this starter or the selected data.

The task repository's docs/VALIDATION.md records the measured scope. Source support for all
backends is not a claim of a competitive jointly trained 39-case model.
