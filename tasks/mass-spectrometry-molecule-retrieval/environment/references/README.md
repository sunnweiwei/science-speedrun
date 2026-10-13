# References: the official-weight GLACIER / clean-MIST target, audits and reproduction scripts

This directory is reference material. The task author ran the official-weight replay below; the
integrator did not re-run it. The integrator's own reproductions (DeepSets and native GLACIER from
scratch) are in `/workspace/baseline/starter/RESULTS.md`. `scripts/` holds the author's
reproduction scripts; five of them are packaged into every native GLACIER submission by
`baseline/starter/glacier/publish.py`.

The scientific baseline to beat is a complete, formula-free hybrid over the
frozen MassSpecGym v1.5 mass-retrieval task:

1. replay the official **GLACIER** no-contrastive-finetuning checkpoint on the
   conditions represented by its released MassSpecGym model; and
2. use the independently trained, strictly held-out-clean **MIST-CF -> MIST**
   reference on the remaining public conditions.

The Fourier-DeepSets package in `/workspace/baseline/starter/` is a fast trainable contract check,
not the reference target. Every method is judged through the same complete
candidate-ID ranking contract and the same 17,556-query denominator.

## Target to beat

The completed frozen replay obtains:

| Metric | GLACIER/MIST hybrid | Direction |
| --- | ---: | --- |
| Hit rate @ 1 | **49.391%** | higher |
| Hit rate @ 5 | 64.189% | higher |
| Hit rate @ 20 | 74.436% | higher |
| MCES @ 1 | 12.179 | lower |

The benchmark's primary improvement value is:

```text
reference_delta_hit_at_1 = submission_hit_rate_at_1 - 0.4939052175894281
```

A positive point estimate provisionally beats the reference. A scientific
claim must additionally have a positive lower endpoint in the task's
query-paired 99.9% bootstrap. This query-level diagnostic does not account for
multiple spectra from one molecule; a new-molecule generalization claim also
needs molecule-grouped uncertainty and appropriate repeated-training evidence.
Against the former 39.8724083% clean MIST
reference, this hybrid gains 9.518 percentage points with interval
[8.345, 10.686].

## Independently trained baseline and literature context

The task-owned scratch GLACIER + trained DeepSets recipe also completed the
full test: **40.322397% / 56.846662% / 70.773525%** Hit@1/5/20 and
MCES@1 **13.169600**, with all 17,556 rankings valid. Its full validation Hit@1
is 35.987441%. Both encoders start randomly; GLACIER uses 99,119 annotated
experimental training spectra and DeepSets uses all 194,119 training spectra.
The optional clean synthetic corpus was not used in this run. Its measured
H200 training and CS A5000/CPU replay establish a reproduced score; single-H100
timing remains unverified. Exact metrics, configuration, branch counts and
artifact hashes are in [task_trained_baseline.json](task_trained_baseline.json).

On 2026-09-19 UTC, a supplemental evaluation combined the same scratch GLACIER
predictions with the existing clean MIST fallback. Trusted rescoring of all
17,556 queries gave **48.285486% / 64.057872% / 74.561404%** Hit@1/5/20
and **12.327666** MCES@1. Integrity checks, merging and scoring took 188.47
seconds on CS with 16 MCES workers and no GPU; Hit@k was available after 13.64
seconds. These timings exclude the historical training and prediction runs.
The published routing predicate was unchanged: 9,954 GLACIER queries and 7,602
MIST queries. Every GLACIER ranking matched the scratch run, and every fallback
ranking matched the official-weight hybrid. Hit@1 increased by **7.963090
percentage points** over GLACIER + DeepSets, leaving a **1.105035-point** gap
to the frozen reference. See [scratch_glacier_clean_mist.json](scratch_glacier_clean_mist.json)
for metrics, integrity checks, timings and artifact hashes.

This is a post hoc combination of frozen cached predictions, followed by a new
full-test score; no training or GPU inference was rerun. Both GLACIER and the
MIST encoder were independently trained, but the MIST-CF formula model and fast
filter use released weights. It therefore does not establish an entirely
from-scratch pipeline, an untouched test-set confirmation, or a packaged
end-to-end runtime measurement. The official 49.390522% target remains fixed.

As checked on 2026-09-13, GLACIER + CF reports **69.95%** in the restricted
simulation-domain mass-retrieval comparison. That is a literature comparison
on a subset, not the full-task reference above. SpecBridge's earlier 84.695%
claim was withdrawn on 2026-03-03 for preprocessing/evaluation problems.
[SOTA_AUDIT.md](SOTA_AUDIT.md) records primary sources, other published methods
including GLMR, and the limits of cross-protocol comparisons.

## GLACIER branch

[GLACIER](https://arxiv.org/abs/2606.29161) treats molecule-to-spectrum
prediction as set prediction. A five-layer Graphormer embeds the molecular
graph; adduct, collision energy, and instrument metadata condition a fixed set
of 200 DETR-style fragment queries. The queries predict multi-bond breakage
events and fragment intensities in one pass. The released MassSpecGym
configuration uses hidden size 256, MAGMa targets capped at six broken bonds,
predicted break cardinality zero through three, four fragment-decoder layers,
three intensity-encoder layers, four intensity-decoder layers, and 150,000
bins over 0--1,500 Da.

For retrieval, the adapter simulates a spectrum for every supplied candidate
and ranks candidates by entropy distance to the observed spectrum. Observed
intensities are max-normalized and square-root transformed; only the 50
strongest positive peaks are retained; spectra are binned at 0.01 Da. Exact
score ties use the content-derived candidate ID, never source-file position.

The base model combines cosine spectrum loss with MAGMa-derived fragment
matching. Its Graphormer and heads are initialized from scratch; there is no
generic molecular backbone pretraining. The optional paper stage then performs
contrastive finetuning against PubChem decoys. The only public checkpoint
available for this replay is the **base no-CF model**: it stores epoch 23,
global step 74,352, and learning rate `4e-4`, strictly loads all 264 tensors,
and contains 15,108,154 parameters. This official-weight reference therefore
reports a checkpoint replay. The separate scratch result above records our
independent GLACIER training.

### Published domain and local result

The official source selects `simulation_challenge=True`: `[M+H]+`, finite
collision energy, and an Orbitrap or QTOF annotation. It contains 99,341 train,
9,734 validation, and 9,954 test spectra. A set-equality audit proves that the
task's public-field predicate selects exactly those identifiers. The broader
frozen test has 17,556 queries, so this branch covers 9,954 (56.70%) and cannot
legitimately claim the paper score over the full task.

On the exact 9,954-query published domain, the local replay obtains
67.490%/84.459%/91.682% exact candidate-ID Hit@1/5/20. The paper reports
69.69%/86.58%/93.34% for the base
model and 69.95%/86.47%/93.52% after contrastive finetuning under its own
candidate-identity conventions. The protocol audit separately records exact
candidate ID, exact InChIKey, and 2D InChIKey so stereochemical duplicates
cannot silently change the comparison.

The full eligible validation replay is within 0.405/0.311/0.100 percentage
points of the paper's no-CF values, while the eligible test replay is lower by
2.200/2.121/1.658 points. Exact candidate, exact InChIKey, and 2D InChIKey
scores are identical locally, and the post-paper source changes between the
GLACIER script release commit and the pinned commit do not touch model or
retrieval code. Because the public artifacts do not preserve the paper-run
ranking or a digest of its materialized mass-candidate table, the remaining
test difference is reported rather than retroactively tuned away. This is a
faithful official-weight/method replay on the task's pinned candidate contract,
not a claim of byte-for-byte recovery of the paper's unpublished run.

GLACIER's fixed atom vocabulary intentionally rejects some unusual candidate
structures. The scorer preserves every such candidate in the output and gives
it the deterministic worst distance, matching the upstream missing-prediction
semantics without dropping a query. The full audit reports
18,727 model-unsupported candidate occurrences and zero missing predictions
for model-supported candidates. All 9,734 eligible validation targets and all
9,954 eligible test targets are model-supported.

## Strict-clean MIST fallback

The remaining 7,602 test queries use the former formal reference unchanged.
MIST-CF enumerates and ranks precursor formulae from precursor mass and the
experimental spectrum. A task-trained MIST vFRIGID encoder predicts a
Morgan4096 fingerprint under each of the top five predicted formulae, combines
them by element-wise maximum, binarizes at the validation-frozen threshold
`0.22049389779567719`, and ranks the unchanged mass-compatible pool by binary
Tanimoto similarity. No ground-truth formula is used and predicted formulae do
not filter candidate pools.

The encoder was trained from scratch on all 194,119 experimental train spectra
plus 2,333,485 ICEBERG simulations after removing 15 held-out-structure rows
and 10 RDKit-invalid rows. The 150-epoch seed-1 run took 18:54:08 on one RTX PRO
6000 Max-Q; validation selected epoch 128. Its 114,206,138-byte exported weight
has SHA-256
`30715dbd198bb1d0d776a5dcb9d37488b8d5480c243a1572d62ed27b73e92c6b`,
and all 59 tensors exactly match the selected EMA checkpoint.

As a standalone full-task method, clean MIST obtains 39.8724083%/51.4866712%/
63.1408066% Hit@1/5/20 and MCES@1 15.1668375. Its ranking SHA-256 is
`78a781884f442039c66963eeab7187f27645440e644d3088f85dbb81e8fd7f3d`.
The complete independent training and formula-stage audit remains in
`REPRODUCTION.md`; it is retained as both provenance and the hybrid fallback.

## Reproduction lock

- MassSpecGym v1.5 data revision:
  `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`.
- Raw `MassSpecGym1.5.tsv` SHA-256:
  `50cfdd1d22f79543c59555f9ce43c6893bd788a19a42b21fb4e2e3a54673c06a`.
- GLACIER source: `coleygroup/ms-pred` commit
  `ed8311f22958cb37f055b663b5f56c5c77a2ee33`, MIT licensed.
- GLACIER archive: 181,602,669 bytes, SHA-256
  `7e0fa24a19163513fe7cbeea3bf51a48cdd33bf2dbe8ac7a0457cbbf28d1258b`.
- Extracted no-CF checkpoint: 181,602,423 bytes, SHA-256
  `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`.
- Upstream GLACIER labels: 119,029 rows, SHA-256
  `0b29f4b150fa043f2ea382df3d7a02ddfdf5dc9baee9790299b3890d3582f244`.
- MIST-CF commit: `71c820b43825e0b0a58680a3690f29a7352290f4`.
- MIST vFRIGID commit: `5e537525a9ae26208efcf882a07e3b8df2413c48`.

The GLACIER source is MIT. Its separately hosted checkpoint has no independent
license metadata in the download, so the task records its digest and does not
redistribute it. Obtain the official artifact through the preparation helper
and seek author clarification before bundling the weight in another image.

## Validation, controls, and determinism

Entropy scoring was selected on a diverse 100-query validation probe before
full replay. The complete 9,734-query eligible validation score is
70.095%/86.891%/93.240% Hit@1/5/20. Test labels were not accessed until this
configuration and its
candidate-identity audit passed.

Replacing every observed test spectrum with one constant two-peak spectrum
drops the full hybrid to 3.139%/6.106%/14.496% Hit@1/5/20 and MCES@1 26.451.
The existing candidate-order null is 0.256%/2.820%/10.338%. These controls
show that the formal score is not reproduced by precursor metadata, pool
composition, or answer-independent ordering alone.

The formal complete ranking SHA-256 is
`b703afcdd0b543486d54b36b79e096de3259b25dbf0e29ebfb876b5d19e7a8c6`. A
second independent four-shard scoring pass and strict merge produce the same
hash; the files are byte-identical. Recovery from upstream DataLoader
host-memory pressure used only public `(query ID, candidate InChIKey, collision
energy)` keys. The detailed failure, recovery, support, and label-access ledger is in
`GLACIER_REPRODUCTION.md`.

## Compute and runnable replay

These are **operator-side** commands for the frozen official-weight/MIST
comparison. The agent image now contains native GLACIER source/runtime and
train/validation-only MAGMa supervision for the scratch recipe in
`/workspace/baseline/starter/glacier/README.md`. It does not contain reference weights or
the external MIST runtime. These reproduction commands do not authorize
pretrained initialization. The completed scratch score is recorded above;
matched-H100 training/replay remains a separate acceptance item.

The construction replay used eight parallel P100 shards. The initial large
feature-cache jobs produced usable predictions for roughly 94--98% of every
shard before a 64 GiB host-memory cgroup OOM; answer-free set-difference jobs
completed the remainder with batch size 2, two workers, a 64-entry feature
cache, prefetch factor 1, and about 5.1--5.3 GiB process RSS. Total construction
cost was approximately 33.1 P100 GPU-hours including retained partial
work, diagnostics, and one cancelled insurance retry. This is inference, not
GLACIER training. A single-H100 six-hour replay has not been directly timed and
must not be inferred from the parallel P100 wall clock.

The verifier's six-hour envelope includes scoring: `main_verifier.py` defaults
to a five-hour model-inference timeout, followed by up to one hour waiting for
the trusted grader. Full package replay must be measured against both limits.

Prepare and verify the immutable operator-side assets:

```bash
bash environment/references/scripts/prepare_glacier_reference_assets.sh ASSET_ROOT
(cd ASSET_ROOT/ms-pred && uv sync --extra cu124)
```

The formal runner requires the independently trained clean MIST weight and
checks its digest rather than silently substituting the weaker released weight:

```bash
export MASS_SPEC_GLACIER_PYTHON=ASSET_ROOT/ms-pred/.venv/bin/python
export MASS_SPEC_MIST_CHECKPOINT=/absolute/path/to/clean/spectra_encoder.pt
bash environment/references/scripts/run_glacier_reference.sh \
  INPUT_ROOT OUTPUT_ROOT ASSET_ROOT
```

The runner uses `MASS_SPEC_GLACIER_PYTHON` for both task adapters by default;
set `MASS_SPEC_REFERENCE_PYTHON` only if the MIST branch needs a separate
compatible interpreter.

`run_glacier_reference.sh` defaults to H100-oriented batch size 16, two loader
workers, feature-cache budget 64, and prefetch factor 1. All are overridable by
`MASS_SPEC_GLACIER_*` variables. The runner consumes only public contract input
and emits `OUTPUT_ROOT/rankings.jsonl`, one complete candidate permutation per
query. `resume_glacier_candidates.py` and the scorer's `--exclude-ranking`
and `--query-id-file` options support answer-free recovery and query-disjoint
parallel scoring; `merge_partial_rankings.py` rejects overlap, unknown queries,
or incomplete permutations.

## What an agent should explore

The baseline is deliberately strong but leaves scientifically useful paths:

- reproduce and improve contrastive finetuning with train-only hard negatives;
- extend one spectrum simulator across sodium adducts and missing instruments,
  eliminating the query-level fallback;
- train a spectrum/molecule dual encoder and use GLACIER only as a reranker;
- improve collision-energy conditioning, spectrum normalization, candidate
  caching, or fragment supervision;
- distill forward simulation into a verifier-efficient candidate scorer; or
- build a single general model that transfers across all public conditions.

The same fixed model/package must handle every query. Validation may select
models and hyperparameters; validation/test labels, true test formulae,
candidate order, and popularity metadata may never train, filter, or route the
model. External NPLIB1 or CASMI transfer is valuable for a broader scientific
claim, but it is not mixed into this task's frozen grader.
