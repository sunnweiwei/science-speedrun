# Native WN2 starter: one complete training run

Run inside the task environment on four GPUs:

```bash
python /workspace/baseline/starter/train.py \
  --data /workspace/data/training \
  --output /workspace/research/training-run \
  --submission /workspace/submission \
  --seed 0
```

This initializes the full model randomly, trains the eleven stages in order,
retains parameters and Adam state between stages, and exports a portable
`infer.sh` package. The supplied `schedule.json` fixes 580 updates:
410, 102, 32, 8, 4, then four updates each for AR3–AR8. You do not need to run
performance measurements, GPU diagnostics, or a second seed before training.
`schedule.json` caps one run at 24 hours of training plus 30 minutes of packaging
(`training_cap_seconds`, `checkpoint_reserve_seconds`); a run that has not finished its
curriculum by then is not exported. The composed measured estimate is 87.51 RTX PRO 6000 GPU-hours (21.88 hours on four
GPUs), based on the original prefix through AR2 and the extended continuation.
This is a reference measurement, not a GPU-model requirement or a runtime
guarantee on other hardware. Hardware-specific execution tuning is allowed
within the training budget; preserve the training objective and effective batch.

Progress and checkpoints are written under the run directory. To continue an
interrupted run, repeat the command with `--resume`, the same seed and paths.
The training time limit includes time since the run first started, including
interruptions. Data provenance, finite gradients and checkpoint consistency
checks remain part of normal training. A failed or partial curriculum is not
exported as a completed model.

The environment compiles kernels as needed. Compiler caches and autotuning
files are local runtime artifacts, not an agent profiling workflow. The runner
uses the model Python environment; it does not require a job scheduler.

After training, follow `/workspace/evaluation/README.md` for visible development
scores. Scoring is a separate action and test truth is not available to this
training entry point. The independent verifier evaluates the submitted model.
