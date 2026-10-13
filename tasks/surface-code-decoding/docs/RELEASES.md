# Release history

Moved from the README when the task adopted TASK_STANDARD.md (paths predate the move: `/data` is
now `/workspace/data`, `/workspace/code` is now `/workspace/baseline/alphaqubit`).

## Release v1.1: no trained weights for agents

From v1.1 the development archive ships no trained weights. The reference
AlphaQubit's code is in the image (`/workspace/code/baselines`, recipe in
`environment/baseline/alphaqubit/reproduction/`), and `/data/reference/results` gives its per-shot validation
predictions, validation metrics, aggregate test score and measured training cost;
`/data/reference/package-template` is its submission layout without weights.
`baselines.alphaqubit_train` reads the development release directly, and
`python -m qecbench.package` turns frozen trained groups into a submission. The
reference's baseline cost is ≈200 GPU-hours on RTX PRO 6000 Blackwell GPUs (up to 8
at once, 47.9 h wall including queueing). The v1.2 budget is described below.

Integrator replay of the frozen reference in the native Docker topology on H100:
P=0.7550746, score 0.85834 (all 3.25M shots; single-GPU inference 31 min on a
shared GPU). The verifier and grader archives were rebuilt from Zenodo and match
the pinned bytes.

## Release v1.2: budget for the from-scratch reference recipe

The agent budget is **8 H100 GPUs × 48 hours** (384 allocated GPU-hours).
`solution/solve.sh` schedules ten independent specialists, one per visible GPU,
with an 18-hour pretraining cap and a fine-tuning cap of two hours per specialist.
With eight visible GPUs the script runs two waves: approximately **36–40 hours**,
plus packaging and validation overhead. This is a scheduling estimate based on
our measured RTX PRO 6000 runs (about 180 pretraining + 20 fine-tuning GPU-hours),
not a completed H100 training reproduction or a guaranteed accuracy result.
Fine-tuning may finish earlier at its step limit; faster GPUs do not automatically
shorten the time-capped pretraining stage. Dividing 200 GPU-hours by eight would
underestimate this script's wall time.

The budget allows one full unchanged reference recipe with roughly 8–12 hours
of margin, not repeated full hyperparameter searches. The estimate assumes all
eight GPUs are available together and the existing microbatch/runtime fits;
with four GPUs the same wave scheduler needs about 54–60 hours, and with two
about 90–100 hours. Agents can use less compute or choose another decoder.
Datasets, submission protocol, fixed score anchors and the one-GPU verifier
budget are unchanged. See the [deployment follow-up](../environment/baseline/alphaqubit/reproduction/deployment-followup.md).

