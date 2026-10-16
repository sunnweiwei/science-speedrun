# Development evaluation

Evaluate one frozen encoder on AWF classification and LFMC moisture regression.
Use downstream training labels only to fit the probes, and validation labels to
select probe settings and compare candidate encoders. The encoder and its
training heads must not learn from either task's labels.

The development data layout is described in [the data guide](../data/README.md):
`/workspace/data/eval/{awf,lfmc}/` contains prepared `inputs.h5` and `labels.json`
with the complete public training and validation populations. The operator must
provide this data before these commands run. Private final-test labels belong
only to the separate verifier and are not mounted in the agent workspace.

The commands below use the installed runtime and tools under
`/workspace/evaluation`. They do not download data or require cluster-specific
launch scripts. Run feature extraction on the allocated GPU, preserving its
device visibility. Preparation, probes, raw features and comparisons run on CPU.

## Use the supplied validation view

Use a fresh validation directory for each comparison. The exported trained and
matched-random packages will live under its `packages/` directory. Link its
`eval/` directory to the operator-prepared read-only view; this reuses the same
inputs, labels and population manifest without copying or preparing them again.

```bash
set -euo pipefail
OLMO_PY=/opt/olmoearth/.venv/bin/python
OLMO_EVAL=/workspace/research/validation
OLMO_NAME=starter-large-seed12536
OLMO_RANDOM=random-large-seed12536
export PYTHONHASHSEED=0 OLMO_INFERENCE_HASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for OLMO_TASK in awf lfmc; do
  test -s "/workspace/data/eval/$OLMO_TASK/inputs.h5"
  test -s "/workspace/data/eval/$OLMO_TASK/labels.json"
done

test -s /workspace/data/eval/manifest.json
mkdir -p "$OLMO_EVAL"
ln -sT /workspace/data/eval "$OLMO_EVAL/eval"
```

The complete view has 1,115 training / 344 validation AWF rows and
31,175 / 8,262 LFMC rows, with no test rows. Reuse this same population across
the trained encoder, random encoder and raw-feature baseline. A reduced view
is a diagnostic, not a comparable full-validation result.

`screening.py prepare --site-fraction 1` is an operator preparation tool for the
original release layout. It has already been applied before mounting this view;
do not run it again on the filtered workspace data, whose labels retain original
row indices. The data guide describes preparation outside the agent workspace.

## Replay the exported packages and fit probes

The [baseline export instructions](../baseline/starter/README.md#export-research-packages)
produce both packages under the paths above. Check that they are complete:

```bash
for OLMO_MODEL_NAME in "$OLMO_RANDOM" "$OLMO_NAME"; do
  for OLMO_FILE in encoder.pt config.json infer.py provenance.json LICENSE-OlmoEarth; do
    test -s "$OLMO_EVAL/packages/$OLMO_MODEL_NAME/$OLMO_FILE"
  done
done
```

If the candidate architecture or seed changes, update the random control to
match. Use the same selected encoder package for both tasks. Packages follow
[the delivery contract](../contract.md); training heads and optimizer state do
not belong in the exported package.

```bash
for OLMO_MODEL_NAME in "$OLMO_RANDOM" "$OLMO_NAME"; do
  for OLMO_TASK in awf lfmc; do
    "$OLMO_PY" /workspace/evaluation/replay.py \
      --model "$OLMO_EVAL/packages/$OLMO_MODEL_NAME" \
      --input "$OLMO_EVAL/eval/$OLMO_TASK/inputs.h5" \
      --output "$OLMO_EVAL/features/$OLMO_MODEL_NAME-$OLMO_TASK"
    CUDA_VISIBLE_DEVICES='' "$OLMO_PY" /workspace/evaluation/probes.py \
      --features "$OLMO_EVAL/features/$OLMO_MODEL_NAME-$OLMO_TASK/features.npz" \
      --labels "$OLMO_EVAL/eval/$OLMO_TASK/labels.json" --task "$OLMO_TASK" \
      --output "$OLMO_EVAL/results/$OLMO_MODEL_NAME-$OLMO_TASK.json" \
      --validation-predictions \
      "$OLMO_EVAL/results/$OLMO_MODEL_NAME-$OLMO_TASK-predictions.npz"
  done
done
```

`replay.py` uses its sibling `namespace.py` and Bubblewrap to expose only the
declared runtime, exported package, unlabeled input file and empty output
directory to inference. It excludes labels, training data and network access.
The environment must support the required user namespaces. Labels enter the
separate probe process afterward. Where the workspace container cannot create
namespaces (Bubblewrap reports "Creating new namespace failed" or "Failed to make
/ slave"), add `--no-namespace`: the package runs through the same entry point with
a clean environment but without isolation. The verifier always uses the namespace,
so a package must not rely on anything outside the contract.

Keep the inference batch size at the adapter's default of **16**, hash seed
**0**, and the same GPU type across paired models. Preserve input normalization,
observation masks, temporal handling and spatial pooling. AWF uses standardized
logistic regression with C in `[0.001, 0.01, 0.1, 1, 10, 100]`; LFMC uses
standardized ridge regression with alpha in
`[0.01, 0.1, 1, 10, 100, 1000, 10000]`. Standardization and estimator fitting use
training rows only; validation selects the probe parameter. The reports retain
the complete search, secondary metrics and constant-predictor baselines.

## Raw features and paired comparisons

```bash
for OLMO_TASK in awf lfmc; do
  CUDA_VISIBLE_DEVICES='' "$OLMO_PY" /workspace/evaluation/raw_features.py \
    --input "$OLMO_EVAL/eval/$OLMO_TASK/inputs.h5" \
    --output "$OLMO_EVAL/features/raw-$OLMO_TASK/features.npz"
  CUDA_VISIBLE_DEVICES='' "$OLMO_PY" /workspace/evaluation/probes.py \
    --features "$OLMO_EVAL/features/raw-$OLMO_TASK/features.npz" \
    --labels "$OLMO_EVAL/eval/$OLMO_TASK/labels.json" --task "$OLMO_TASK" \
    --output "$OLMO_EVAL/results/raw-$OLMO_TASK.json" \
    --validation-predictions "$OLMO_EVAL/results/raw-$OLMO_TASK-predictions.npz"
done

CUDA_VISIBLE_DEVICES='' "$OLMO_PY" /workspace/evaluation/compare_seeds.py \
  --data-root "$OLMO_EVAL" --pairs "$OLMO_RANDOM:$OLMO_NAME" \
  --raw-baseline raw --output "$OLMO_EVAL/paired-comparison.json"
```

Raw features use the same image normalization and probe protocol without a
learned encoder. `compare_seeds.py` checks paired package provenance and reports
validation gains with spatial/site bootstrap intervals. One pair measures
uncertainty conditional on one training seed; it is not independent-seed
replication. Multiple pairs require distinct initialization/data seeds and the
same recorded scientific recipe. Retain the view manifest, exported provenance,
feature files, reports and validation predictions with the comparison.

## Validation metrics and the final verifier

During development report **AWF validation accuracy** and **LFMC validation MAE**,
alongside AWF balanced accuracy and LFMC RMSE/R². The starter's reported
77.034884% AWF accuracy and 21.459503 LFMC MAE are validation results from one
adaptively selected recipe and training seed. They are not a final verifier
reward or evidence of untouched-test generalization.

The separate verifier independently replays the delivered package and uses the
existing reward:

```text
LFMC test skill = max(0, 1 - LFMC test MAE / max(constant_test_MAE, 1e-12))
reward = 0.5 * (AWF validation accuracy + LFMC test skill)
```

Accuracy is a fraction in this formula. LFMC probe settings are selected on
validation before final-test scoring; the constant predictor is fitted on
training labels; `constant_test_MAE` is its test MAE. Final-test labels remain outside the agent workspace, and
final scoring follows checkpoint and hyperparameter selection. None of the
development commands above requests `--final-test` or invokes the final grader.
Independent replay checks the delivered artifact; it does not turn earlier
public validation development into a blind holdout study.
