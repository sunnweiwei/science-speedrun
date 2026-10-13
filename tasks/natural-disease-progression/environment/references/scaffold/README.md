# Scaffold interface

`/workspace/references/scaffold/` contains an architecture-neutral training/inference scaffold, not a default model.

- `model.py` owns only `build_model(config, metadata)` and the returned module's `forward(batch)` computation. Implement the architecture there.
- `data.py` reads complete cutoff-bounded histories and owns the replaceable Dataset, fitted preprocessor, and collate seams used by both training and inference.
- `train.py` organizes optimization and saves `model_state`, plain configuration, and plain metadata in one checkpoint. Implement the training loader and loss at the marked `NotImplementedError` sites.
- `predict.py` independently restores that checkpoint, reconstructs preprocessing and the model through the shared builders, and writes keyed predictions.
- `config.json` separates `model`, `data`, and `training`. Its values are placeholders, not a historical best configuration.
- `infer.sh INPUT_DIR OUTPUT_DIR` keeps the external verifier interface unchanged.

Start by implementing the explicit seams in `data.py`, `model.py`, and `train.py`; they fail loudly until supplied. Keep fitted vocabularies, imputation values, and normalization values inside the checkpoint's JSON-like `metadata["preprocessing"]`. Do not store Python objects there: inference deliberately uses `torch.load(..., weights_only=True)`.

No sequence length, event sampling, vocabulary, embedding size, pooling rule, loss, optimizer, or fold is imposed. The default history iterator returns every public event. Fit every learned preprocessing artifact on `data/train` and package it with the final submission.

Minimal flow after implementing the seams:

```bash
python /workspace/references/scaffold/train.py --input /workspace/data/train \
  --config /workspace/references/scaffold/config.json --checkpoint /workspace/research/scaffold/checkpoint.pt
/workspace/references/scaffold/infer.sh /workspace/data/public/validation /workspace/research/candidate
```

The final package must be a jointly trained neural model with learned patient-specific predictions at all 13 horizons. Include model and inference code, a training entry point, frozen configuration, trained weights, preprocessing artifacts, and a README describing the structural change from the reference and the evidence used to choose it.

Neural examples are the verified baseline `/workspace/baseline/temporal-gru` and the unverified reference `/workspace/references/delphi`. Copy code you intend to change into `research/`; copy only the final reproducible runtime into `submission/`. The longitudinal example is runnable as an independent cache/train/predict workflow but its 512-event policy, GRU dimensions, pooling, loss, and data folds are merely reference decisions.

Validate predictions locally with:

```bash
submission/infer.sh /workspace/data/public/validation /workspace/research/candidate
python /workspace/evaluation/score.py \
  --input /workspace/data/public/validation \
  --trusted /workspace/data/trusted/validation \
  --normalization /workspace/data/trusted/risk_normalization.parquet \
  --predictions /workspace/research/candidate/predictions.parquet \
  --output /workspace/research/candidate-score
```
