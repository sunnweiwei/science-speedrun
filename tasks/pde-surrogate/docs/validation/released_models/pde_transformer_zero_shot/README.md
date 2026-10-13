# Reference: released PDE-Transformer checkpoints, zero-shot

Replays the six released supervised checkpoints (MIT, pinned revision
`d5889aab16b1fe82e82940b612d54a8f0773e542`) under the task protocol without
any training, as the pretraining-corpus reference. It reproduces upstream's
inference path exactly: the mixed-channel model takes one normalized frame with
two channels (zero-padded), conditioned on the PDE class index, and is rolled
out one step at a time. Normalization uses the `norm_fields_sca_*` buffers of
the corpus training files, bundled by `build.py`.

Two departures from the paper's evaluation, both documented in the numbers:
the rollout starts from the task's fourth frame instead of the first (the
model receives only that frame), and the held-out simulations are the task's
split (`sims 500–599` of the full release, identical to the paper's range, and
the upstream `_test` files). The transfer datasets (The Well, PDEArena) were
not in the checkpoints' training data; their zero-shot numbers are reported
for completeness only.

```bash
python3 build.py --data "$TASK_PRIVATE_DIR" --source ape2d-full
PDET_CKPT_DIR=/path/to/pde-transformer PDET_VARIANT=mc-b bash infer.sh INPUT_DIR OUTPUT_DIR
```

Requires `pdetransformer==0.1.11` with `transformers<5`.
