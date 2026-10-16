# Verified specialist references on Research v2

These are **per-case comparison baselines**, not the agent's required shared
submission and not a claim of full-paper convergence. The verified table and
machine-readable results.json record all 39 cases: see `verified/` (`/workspace/references/results` in the image)
in the repository, or /workspace/references/results in the development image.

## Runnable in the agent workspace

The development image exposes only the current reference subset at
`/workspace/references/specialists`. Author source is pinned at
`/opt/references/hydrogym-author`; no Docker-in-Docker or host path is needed.

```bash
# Example matching the current public PPO recipe; choose a fresh output path.
CUDA_VISIBLE_DEVICES=0 bash /workspace/evaluation/native_runtime.sh maia python \
  /workspace/references/specialists/research_train.py \
  --case Cylinder_2D_Re200 --algo PPO --recipe public --seed 42 \
  --steps 20000 --eval-every 4000 --save-every 400 \
  --data /data --output /workspace/research/reference-cylinder-001
```

Use `jax` for Channel/Kolmogorov and `jax-fluids` for Nozzle. Bind exactly the
case's frozen solver-GPU count (three cases require two distinct GPUs); the
learner and solver share that allocation. Select the algorithm/recipe/budget
from the published results JSON, not from unrelated historical scripts.
The remaining slow cases use 4,000 controls; the table records actual budgets
and selected checkpoint steps separately.

The trainer starts with zero physical control and random-policy checks, trains
from initialization using the pinned author factory, exports checkpoints every
400 controls, and evaluates at the declared interval. It preserves source
physics, native bounds, observation pipeline and finite-horizon critic bootstrap.
Native reward is a **training signal**, not the reporting metric.

To independently evaluate an exported comparison checkpoint:

```bash
bash /workspace/evaluation/evaluate_native.sh --data /data \
  --submission /workspace/research/reference-cylinder-001/submission-0020000 \
  --case Cylinder_2D_Re200 --output /workspace/research/reference-replay-001
```

The actor export is relocated code/weights with deterministic inference.
It is still a specialist; exporting it does not make it a valid shared-model
submission. The agent's shared starting point is `/workspace/baseline/starter/train_native.py`.

## Reproducibility and limitations

The verified results.json includes exact learner configs, seeds, source versions,
best-model and independent-replay hashes, the run's evaluator identity, and
paper figure provenance. It excludes raw CFD data, trajectories and checkpoints.
Best means the best physical metric among positive-step development checkpoints,
using visible evaluation conditions; it is not blind-test selection.
Zero physical control is not a random model or the gust's undisturbed target.
One fixed initialization does not provide confidence intervals.

Public code is adapted through documented TorchRL API compatibility changes,
not asserted to be the undisclosed final paper setup. Some recipes change
initial policy scale; this is recorded, not silently pooled with public defaults.
Historical long-training paper-profile results are separate entries in JSON.

The repository retains dated audits, old runner scripts and reconstruction code
for provenance. They are not copied as the active agent reference and are not
instructions to train on excluded cavity resources. Do not run old local-path
schedulers as part of this Harbor task.
