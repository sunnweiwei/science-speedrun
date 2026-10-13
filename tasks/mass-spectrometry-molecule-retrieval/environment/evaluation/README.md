# Development and verifier evaluation

## Metrics

For each spectrum, the correct molecule is present exactly once in its candidate
pool. The evaluator computes:

- `hit_rate_at_1` (primary, higher is better),
- `hit_rate_at_5` (higher is better),
- `hit_rate_at_20` (higher is better), and
- `mces_at_1` (lower is better).

Hit rate is the fraction of all query spectra whose correct candidate occurs in
the first k ranked IDs. MCES@1 is MyopicMCES with the upstream default threshold
15 and stronger bound. Missing or invalid queries remain in the denominator,
miss at every k, and receive MCES distance 100. The normalized benchmark score transforms only Hit@1; the secondary metrics
are reported separately.
The grader solves each unique `(true SMILES, predicted SMILES)` pair once and
then applies that distance to every matching spectrum; this is exactly
equivalent to per-spectrum solving and only removes duplicate computation.

The paper reports hit rates as percentages; raw Hit@k metrics use fractions.
The canonical verifier reward is the separate normalized `reward` in [0, 1] (`reward.json`).

## Reference comparison

The formal target is the complete GLACIER-no-CF/strict-clean-MIST hybrid
replayed on this exact frozen input at
`hit_rate_at_1 = 0.4939052175894281`. The report includes
`reference_delta_hit_at_1`, defined as submission Hit@1 minus
that value, and a point-estimate `beats_reference` flag in the **test verifier
report**. The validation tool lists the test reference only as non-comparable
context and does not compute a cross-fold improvement. GLACIER covers exactly
9,954 published-domain queries and clean MIST covers the other 7,602; neither
branch receives a ground-truth formula or a changed candidate set. The former
standalone clean MIST target was 0.39872408293460926. See
`/workspace/references/README.md` for the condition-domain rule, model and
weight lock, controls, replay hashes, and the stronger paired-bootstrap
criterion used for a publication claim.

## Visible validation

Validation labels and candidate pools are available under `/workspace/data`. Convert the
validation rows into the frozen inference layout:

```bash
python /workspace/evaluation/stage_validation.py \
  --input-root /workspace/data --output-root /workspace/research/validation-input
bash /workspace/submission/infer.sh \
  /workspace/research/validation-input /workspace/research/validation-output
python /workspace/evaluation/score.py \
  --input-root /workspace/data \
  --rankings /workspace/research/validation-output/rankings.jsonl
```

Add `--mces` to compute the slower structural-distance diagnostic. The hit-rate
logic matches the trusted grader but validation labels are not private.

The report also includes adduct, instrument, collision-energy, candidate-pool
size and routing-domain strata. Its 95% interval resamples molecules using the
supplied InChIKey connectivity block, while retaining spectrum weights in the
primary statistic. Correlated spectra are not counted as independent molecules.
Use `--baseline-rankings BASELINE.jsonl` for a paired interval on the same
validation query set, and `--output REPORT.json` to archive the result.
Intervals from small subsets are exploratory; they do not replace full-fold
model selection or independent agent trials.

For a cheap interface smoke test, pass `--sample-size 32 --seed 7` to
`stage_validation.py` (or use `--limit 512` for a prefix probe)
and pass `--query-input /workspace/research/validation-input` to `score.py`. Always label
the resulting metrics as subset diagnostics rather than validation results.

## Independent test replay

The verifier runs the package as an unprivileged user with no network. It sees
only test spectra and public candidate pools, then seals `rankings.jsonl` and
kills remaining submission processes. A separate grader receives that immutable
snapshot and the label file. The grader never imports or executes submission
code.

Because MassSpecGym's source dataset is public, this boundary protects the
evaluation process but does not turn the official test fold into an unknown
dataset. Benchmark rules and review remain necessary to prevent memorized answer
tables or test-fold training.

## Paired reference comparison

Hit@1 outcomes are paired per query, with uncertainty clustered by molecule.
For an operator-side scientific comparison,
retain the frozen reference ranking and use the private labels to run:

```bash
python /workspace/evaluation/paired_bootstrap.py \
  --labels PRIVATE_ROOT/labels.jsonl \
  --baseline REFERENCE_RANKINGS.jsonl \
  --challenger SUBMISSION_RANKINGS.jsonl \
  --output paired-bootstrap.json
```

The default is a deterministic 100,000-resample 99.9% molecule-cluster paired
percentile bootstrap. It retains spectrum weights in Hit@1, groups private
`true_smiles` by RDKit InChIKey connectivity block, and records the RDKit version.
Canonicalization can give a slightly different group count from the upstream
supplied keys; report the observed count. Use `--unit query` only to reproduce
the historical query-bootstrap reports. A positive point estimate sets the verifier's provisional
`beats_reference` flag; a scientific claim that the model beats the reference
requires the reported lower interval endpoint to be above zero. This utility is
operator-side because neither test labels nor reference predictions are exposed
to the agent.

## Normalized benchmark score

The canonical Harbor reward is `{"reward": S}`, using fixed, versioned empirical
reference anchors for full-test Hit@1 (`P`):

```text
L = 0.04004329004329004  # repaired Fourier-DeepSets acceptance replay
U = 0.4939052175894281   # strongest reproduced full-test GLACIER + clean MIST reference
S = clip((P - L) / (U - L), 0, 1)
```

`U` is a reproduced reference, not a theoretical optimum. Scores above `U`
saturate at 1; `report.json` preserves raw Hit@1/5/20, MCES@1, the unclipped
score, saturation status and the score version. These raw metrics distinguish
further improvements. The lower reference scores 0 and the upper scores 1.
The independently trained scratch GLACIER + DeepSets baseline remains a
separate result; do not confuse it with the official-weight upper reference.
Using the latter as an anchor does not authorize pretrained weights for agents.
Bounds are frozen in `environment/evaluation/score_bounds.json`; changing them
requires a new score version, never per-run or candidate-dependent calibration.
Validation reports apply the same transform only as a development diagnostic;
they cannot establish improvement over a test-fold reference.
Invalid submissions receive score 0. Infrastructure/setup/scorer failures do
not produce a reward. This scales reference-relative performance, not task
scientific importance or equal research difficulty.
