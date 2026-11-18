# Frozen molecular-retrieval contract

## Submission package

Maintain a complete package at `/workspace/submission`:

```text
submission/
  serve.sh              # recommended persistent single-spectrum streaming server
  infer.sh              # legacy per-batch entrypoint (fallback when serve.sh is absent)
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

`model_manifest.json` contains these top-level fields (with optional `"streaming_entrypoint"`):

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

Checkpoint, config, and optional `streaming_entrypoint` paths are relative and
must stay inside the package. `format` may be `pytorch` or `safetensors`. This release authorizes no pretrained
assets, so `pretrained_assets` must be `[]`; listing an external source does not
authorize its use. The verifier rejects a nonempty declaration. The pinned
native runtime `/opt/glacier/bin/python` is supplied in both environments;
model source and weights belong in the submission.

## Invocation

Produce one ranking for each spectrum independently of all other test queries.
Each frozen candidate pool corresponds to one exact target SMILES, so spectra
that share a pool share the answer. The verifier supports two replay modes:

### 1. Persistent Single-Spectrum Streaming Mode (`massspecgym-retrieval-stream/v2`, Recommended)

When the package provides `serve.sh` (or `predict_server.py`, or declares
`"streaming_entrypoint"` in `model_manifest.json`), the verifier launches **one**
persistent unprivileged (`UID 10001`), network-isolated worker process:

```bash
bash /path/to/submission/serve.sh
```

The worker loads its model once, reads **one newline-delimited JSON request at a
time** from `stdin`, and writes **one newline-delimited JSON ranking response**
to `stdout` (flushing immediately; send any library logs to `stderr`):

- **Single-query request** (`query_id` anonymized, `candidate_pool_id` omitted,
  `candidates` shuffled):
  ```json
  {"query_id":"sq_0000000","mzs":[91.0542,125.0233],"intensities":[0.4,1.0],"precursor_mz":288.1225,"adduct":"[M+H]+","instrument_type":"Orbitrap","collision_energy":30.0,"candidates":[{"candidate_id":"sha256-hex","smiles":"CC(=O)O"}]}
  ```
- **Single-query response** (strictly `query_id` and `ranking`; **no `scores`,
  SMILES, or extra fields**):
  ```json
  {"query_id":"sq_0000000","ranking":["best-candidate-id","second-candidate-id"]}
  ```

The verifier dispatches query `i + 1` only after receiving and validating a
complete candidate-pool permutation for query `i` (blocking all lookahead).
Inference **must be strictly stateless across queries** and **must break equal
internal scores deterministically** using content-derived keys (for example
`candidate_id`, never input candidate order, random tie-breaks, or cross-query
counters/caches). Immediately after all queries finish, the verifier kills all
`UID 10001` processes, purges `/tmp` and IPC state, samples up to 128 queries
(prioritizing repeated/isomeric candidate pools in shuffled order), and verifies
**exact full `ranking` list equality** on a fresh cold-start worker; any
mismatch fails with `StatefulQueryLeakageError` (`score = 0.0`, `reward = 0.0`).
The streaming budget is 18,000 seconds of compute plus startup allowance for the
main pass and cold-start spot-check.

### 2. Legacy Per-Pool Batch Fallback (`massspecgym-per-spectrum-batches/v2`)

When `serve.sh` and `predict_server.py` are absent, the verifier falls back to
splitting the test queries into batches with at most one spectrum per candidate
pool (batch j holds the j-th spectrum, in a fixed seeded order, of every pool
with more than j spectra; the number of batches is the largest pool size) and
invokes the package once per batch, in a fresh process:

```bash
bash /path/to/submission/infer.sh INPUT_ROOT OUTPUT_ROOT
```

Both arguments are absolute directories. Each invocation receives a fresh input
root with only its batch's queries and the pools they reference, a fresh empty
output directory and a fresh `HOME`. Input is read-only. Files written anywhere
else, background processes and System V IPC objects are removed before the next
invocation, and the full test input is not readable. Inference must not fit
parameters, mutate the package, contact a service, or leave background
processes. The rows of all invocations are concatenated and scored together; a
row for a query outside its own invocation's input invalidates the output.

Replay has one H100, 16 CPU cores, 64 GiB RAM and 50 GiB writable storage.
The inference budget is 18,000 seconds of compute plus 40 seconds per invocation
for starting the model. The 373 invocations of the full frozen test have a
32,920-second inference deadline. The overall verifier deadline is 36,520 seconds,
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
- Multiple spectra of the full test set share a candidate pool, but one
  invocation never contains two spectra of the same pool.

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
exactly once, ordered from most to least likely. Break ties on equal internal
scores deterministically using candidate identity (for example `candidate_id`),
never input candidate order or cross-query state. Do not emit unknown IDs,
duplicate queries, duplicate candidates, partial top-k lists, scores, SMILES, or
extra fields. Rows in batch `rankings.jsonl` may appear in any order; streaming
responses must answer the in-flight `query_id` one at a time.

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

The canonical Harbor reward is `{"reward": P, "score": P}`, where `P` is the raw
full-test Hit@1 (`mean(1[ranking[0] == correct_candidate_id])` over all 17,556
test queries). For context, the repaired Fourier-DeepSets acceptance replay reaches
P = 0.04004329004329004 and the strongest reproduced full-test GLACIER + clean
MIST reference P = 0.4939052175894281 (a reproduced reference, not a theoretical
optimum). `report.json` keeps Hit@1/5/20, MCES@1 and, as a diagnostic only, the
earlier reference-normalized score from `environment/evaluation/score_bounds.json`.
The independently trained scratch GLACIER + DeepSets baseline remains a
separate result; do not confuse it with the official-weight reference, and the
reference does not authorize pretrained weights for agents.
Invalid submissions receive reward 0. Infrastructure/setup/scorer failures do
not produce a reward.
