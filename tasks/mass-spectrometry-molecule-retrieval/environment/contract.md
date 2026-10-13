# Frozen molecular-retrieval contract

## Submission package

Maintain a complete package at `/workspace/submission`:

```text
submission/
  infer.sh
  model_manifest.json
  README.md
  model code, configuration, weights, and owned runtime assets
  EXPERIMENTS.md and research/  # campaign audit records; not a scored model input
```

The package may be read at any moment, so every update must switch it atomically. The starter
publishes immutable model generations under `.versions` and switches the root manifest with one
rename; building elsewhere and renaming the whole directory also works.

The package may be relocated and invoked from an unspecified working directory.
Resolve owned files relative to the package, and do not depend on the training
workspace, optimizer state, an undocumented cache, or network access. Symlinks
must remain inside the package.

Keep package files readable and directories traversable by UID 10001. The
verifier supplies system libraries and `/opt/glacier/bin/python`; these declared
runtime paths are allowed. `/workspace/data`, the development workspace, and organizer
paths are unavailable. Research records are archived with the package, but the
automated validity checks require `infer.sh`, `model_manifest.json`, `README.md`
and the declared model assets; report completeness is assessed separately.

`model_manifest.json` contains exactly these top-level fields:

```json
{
  "schema": "massspecgym-retrieval-model/v1",
  "checkpoint": {
    "path": "model.pt",
    "sha256": "64 lowercase hexadecimal characters",
    "format": "pytorch"
  },
  "model_config": "model_config.json",
  "training_provenance": {
    "data": "MassSpecGym v1.5 train fold",
    "pretrained_assets": []
  }
}
```

Checkpoint and config paths are relative and must stay inside the package.
`format` may be `pytorch` or `safetensors`. This release authorizes no pretrained
assets, so `pretrained_assets` must be `[]`; listing an external source does not
authorize its use. The verifier rejects a nonempty declaration. The pinned
native runtime `/opt/glacier/bin/python` is supplied in both environments;
model source and weights belong in the submission.

## Invocation

The verifier invokes the package once:

```bash
bash /path/to/submission/infer.sh INPUT_ROOT OUTPUT_ROOT
```

Both arguments are absolute directories. Input is read-only and output starts
empty. Inference must not fit parameters, mutate the package, contact a service,
or leave background processes.

Replay has one H100, 16 CPU cores, 64 GiB RAM and 50 GiB writable storage.
Inference is limited to 18,000 seconds. The entire verifier has 21,600 seconds,
including input/package validation, inference, sealing and trusted grading.
The label-side grader is a separate CPU-only service and is not available to
the model. Input/package size must fit the replay environment; the agent's
larger development allocation does not increase inference limits.

## Input

`INPUT_ROOT/queries.jsonl` has one JSON object per experimental spectrum:

```json
{"query_id":"MassSpecGymID9999999","mzs":[91.0542,125.0233],"intensities":[0.4,1.0],"precursor_mz":288.1225,"adduct":"[M+H]+","instrument_type":"Orbitrap","collision_energy":30.0,"candidate_pool_id":"sha256-hex"}
```

- `query_id` is unique.
- `mzs` and `intensities` are equal-length nonempty finite arrays, ordered by
  increasing m/z. Intensities are nonnegative and have a positive maximum.
- `precursor_mz` is positive and finite.
- `adduct` is `[M+H]+` or `[M+Na]+` in the pinned release.
- `instrument_type` may be an empty string and `collision_energy` may be null.
- Multiple spectra may share a candidate pool.

`INPUT_ROOT/candidate_pools.jsonl` contains each referenced pool once:

```json
{"candidate_pool_id":"sha256-hex","candidates":[{"candidate_id":"sha256-hex","smiles":"CC(=O)O"}]}
```

Candidate IDs are SHA-256 hashes of the UTF-8 SMILES. Every pool contains 1 to
256 unique candidates in a deterministic, answer-independent order. Candidate
SMILES are legitimate inference inputs. A future conforming input may change all
IDs, spectra, pool sizes, candidate structures, ordering, and batch size.

`INPUT_ROOT/manifest.json` records the schema, source revision, counts, and
SHA-256 of the two JSONL files.

## Output

Write exactly one file, `OUTPUT_ROOT/rankings.jsonl`, with one row per query:

```json
{"query_id":"MassSpecGymID9999999","ranking":["best-candidate-id","second-candidate-id"]}
```

For each query, `ranking` must contain every candidate ID in its referenced pool
exactly once, ordered from most to least likely. Do not emit unknown IDs,
duplicate queries, duplicate candidates, partial top-k lists, scores, SMILES, or
extra fields. Rows may appear in any order.

Missing query rankings are retained in the evaluation denominator and score as
misses at every k. Any invalid row (including a partial/duplicate-candidate
ranking, unknown query, or duplicate query) invalidates the whole output and
all queries score as misses. The verifier never uses candidate position as an
answer signal. A timeout, model-process failure, or invalid package is recorded
as a failed submission with zero hit rates and MCES 100. Task-input, runtime
setup and trusted-scoring failures produce an error report without a reward;
they are not measured model performance.

## Metrics

The primary scientific metric is mean hit rate at 1 over every test spectrum. Hit rates at
5 and 20 are computed identically. MCES@1 is the mean MyopicMCES graph-edit
distance between the top-ranked and correct molecules; lower is better. A
missing/invalid query receives MCES distance 100. There is no hidden weighted
aggregate.

## Reward

The canonical Harbor reward is `{"reward": P}`, the raw full-test Hit@1. For
context, the repaired Fourier-DeepSets acceptance replay reaches
P = 0.04004329004329004 and the strongest reproduced full-test GLACIER + clean
MIST reference P = 0.4939052175894281 (a reproduced reference, not a theoretical
optimum). `report.json` keeps Hit@1/5/20, MCES@1 and, as a diagnostic only, the
earlier reference-normalized score from `environment/evaluation/score_bounds.json`.
The independently trained scratch GLACIER + DeepSets baseline remains a
separate result; do not confuse it with the official-weight reference, and the
reference does not authorize pretrained weights for agents.
Invalid submissions receive reward 0. Infrastructure/setup/scorer failures do
not produce a reward.
