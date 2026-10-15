# Frozen crystal-generation contract

Maintain a complete package at `/workspace/submission` containing:

```text
generate.sh
model_manifest.json
README.md
model code, one checkpoint, configuration, and owned offline assets
```

The package may be read at any moment. Replace it as a whole: assemble it elsewhere on the same
filesystem and swap it in with one rename, as the starter's `publish.py` and `train.py` do.
A package that fails validation, exits nonzero or exceeds the inference limits scores 0.

The verifier relocates the package and invokes:

```bash
bash generate.sh INPUT_ROOT OUTPUT_ROOT
```

The working directory is unspecified. Input is read-only, output starts empty,
and inference has no network. The package may not call the evaluator, train,
modify its weights, or require files from the development workspace.

Inference limits: the package runs as an unprivileged user from a read-only copy, with no
network, the verifier image's pinned runtime (nothing is installed), up to eight visible GPUs
(`CUDA_VISIBLE_DEVICES`), and 24 hours of wall time for the whole request. Output is limited to
20 GiB and 20,100 files. Processes left behind after `generate.sh` exits are killed.

After sealing the generated outputs, the trusted grader relaxes a separate copy
using frozen UMA `uma-s-1` (`omat`), BFGSLineSearch + FrechetCellFilter,
`fmax=0.02` eV/Å, at most 200 steps and 300 seconds per structure. It then
scores those geometries with the pinned LeMat ensemble. The package cannot skip
or modify this step and does not receive pretrained scoring weights. Failed
refinements retain the original candidate; finite unconverged iterates are kept.
All requested samples remain in the denominator. The same relaxation is available
during development through `/workspace/evaluation/relax.sh` and is included
automatically in `/workspace/evaluation/score.sh`.

## Requests

`INPUT_ROOT/requests.jsonl` contains one `crystalgen-request/v0` object per line.

Unconditional example:

```json
{"schema_version":"crystalgen-request/v0","request_id":"unconditional-final","mode":"unconditional","condition":null,"n_samples":2500,"seed":1847}
```

Rules:

- `request_id` is unique and matches `[A-Za-z0-9][A-Za-z0-9_.-]*`.
- `mode` must be `unconditional` and `condition` must be `null`.
- Exactly the six fields shown above are accepted; conditional requests are rejected.
- `n_samples` is an integer from 1 through 10,000 and `seed` is a nonnegative
  integer; booleans are not integers for either field.

Final evaluation uses one 2,500-sample request with a fresh operator-held seed.
It does not depend on held-out structures or target properties. The v0 request
envelope is retained for compatibility with unconditional inputs; the final
suite manifest is `crystalgen-final-suite/v1` and rejects legacy mixed suites.

## Outputs

Write `OUTPUT_ROOT/manifest.csv` with exactly these columns:

```csv
request_id,sample_index,structure_path
unconditional-final,0,structures/unconditional-final-0000.cif
```

There must be exactly one row for every requested `(request_id, sample_index)`
pair and no other rows. `sample_index` spans `0 .. n_samples-1`. Paths are
relative, remain inside the output directory after resolution, are unique, and
point to nonempty regular `.cif` files. Symlinks are not accepted.

The optional `OUTPUT_ROOT/report.json` is diagnostic only and is never trusted
for scoring.

Missing, duplicate, extra, malformed, escaping, or unparsable outputs count as
failed requested samples. Contract-level ambiguity or evidence of tampering may
instead fail verification.

## Model manifest

`model_manifest.json` follows `crystalgen-model/v0`:

```json
{
  "schema": "crystalgen-model/v0",
  "checkpoint": {"path": "model.safetensors", "sha256": "...", "format": "safetensors"},
  "parameter_count": 12400000,
  "one_parameter_set": true,
  "trained_from_scratch": true,
  "training_data": {"name": "MP-20", "split": "train", "source_sha256": "133bab58..."},
  "model_config": {"architecture": "..."}
}
```

Competitive submissions use one `.safetensors` checkpoint. The verifier parses
its tensor shapes and requires their total element count to equal
`parameter_count`, which must be from 1 through 70,000,000. JSON checkpoint
format is reserved for the task's tiny interface smoke baseline. Undeclared
`.bin`, `.ckpt`, `.npy`, `.npz`, `.pt`, `.pth`, or `.safetensors` files are
rejected, as are packages above 2 GiB or 10,000 files.

The checkpoint path must remain inside the package and its SHA-256 digest must
match. Training provenance and disguised transformed-weight reuse still require
code/checkpoint review; those properties cannot be proven from bytes alone.
