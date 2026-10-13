# Human 1 Mb / 11-output compact benchmark protocol

## Inference and release boundary

The current release keeps the checksummed target shards unchanged. A
manifest-only exporter creates `public`, `labels`, and `scoring` beside
`targets`, `metadata`, and `release.json` in one layout.
Public rows contain only IDs and sequence-window coordinates. Labels retain the
target reference and real splice positions. Scoring retains the frozen
8,192 legacy junction index queries and, after the full-split supervision audit,
adds at most one deterministic nonzero query for each available track and
example. The maximum frozen budget is therefore 8,926 strand-aware coordinate
queries per example; no unlabeled track is manufactured as a negative.

The splice position convention is fixed to rows 0 donor-positive, 1
acceptor-positive, 2 donor-negative, and 3 acceptor-negative. Coordinates are
zero-based relative to the 1 Mb input. This follows the upstream implementation
at commit [`eecf3d60234dda0428f5f083d22e518b72487ce0`](https://github.com/google-deepmind/alphagenome_research/tree/eecf3d60234dda0428f5f083d22e518b72487ce0),
specifically `model/heads.py` and `model/splicing.py`.

Inference predicts all splice coordinates from DNA. Usage is emitted for every
base in the central target, while the full junction cube is keyed by four
model-predicted, coordinate-sorted site lists. Trusted grading gathers hidden
usage positions and matches hidden junction coordinates; missing predicted
endpoints score as zero. All eleven families remain in the equal-family mean.
This compact protocol is not the paper's complete evaluation protocol.

## Status and scope

This document specifies the single released data and evaluation protocol.
There is no legacy or reduced-track profile in this task.

The protocol goal is a human-only, AlphaGenome-style benchmark that:

- uses a fixed 1,048,576-bp input sequence;
- preserves the complete official `FOLD_0` fold1 manifest as metadata while
  materializing a fixed, reproducible compact test subset;
- exposes all 11 released AlphaGenome output families;
- remains streamable and reasonably affordable for models with at most 50M
  parameters;
- keeps the final materialized train/validation/test release below 100 GiB;
- permits an official AlphaGenome `FOLD_0` model to be evaluated through the
  same coordinate transforms, masks, track selection, and grader.

Changing the model architecture or training recipe is out of scope for this
dataset/protocol migration.

## Frozen coordinate convention

Each example starts from an official AlphaGenome source interval and anchor.
The example ID is derived from the source version, fold, chromosome, anchor,
and source record identity; it is never derived from materialization order.

- Input span: exactly 1,048,576 bp from the human reference genome.
- Default one-dimensional target span: the official central 196,608 bp.
- Input and target coordinates are stored explicitly as zero-based, half-open
  genomic intervals.
- Chromosome-edge padding follows one documented deterministic rule and is
  included in the input mask.
- Predictions are cropped and transformed from these stored coordinates, not
  from implicit array-center assumptions.
- Contact maps use the complete 1 Mb input-aligned map supplied by the source
  data unless the source-shape audit proves otherwise.

The materializer must first verify the coordinates and DNA hash for the same
record across every required bundle. A mismatch is a hard error and must not be
repaired by positional zipping or silent record deletion.

## Output families and compact representations

The protocol uses a separate representation and metric contract for each
family. It does not force incompatible outputs into a single dense tensor.

| Output family | Source/native form | Compact form | Stored resolution |
| --- | --- | --- | ---: |
| ATAC | dense 1 bp | sum-pooled dense tracks | 128 bp |
| DNase | dense 1 bp | sum-pooled dense tracks | 128 bp |
| TF ChIP-seq | dense | native dense tracks | 128 bp |
| Histone ChIP-seq | dense | native dense tracks | 128 bp |
| RNA-seq | dense 1 bp | sum-pooled dense tracks | 128 bp |
| CAGE | dense 1 bp | sum-pooled dense tracks | 128 bp |
| PRO-cap | dense 1 bp | sum-pooled dense tracks | 128 bp |
| Splice sites | base classification | 1-bp labels or sparse labeled events | 1 bp / event |
| Splice-site usage | sparse signal on sites | positions, values, tracks, masks | event level |
| Splice junctions | splice-site pair tensor | positions plus valid junction pairs | junction level |
| Contact maps | pairwise map | pairwise map, optionally triangular storage | 2,048 bp |

The protocol chooses 128 bp rather than 256 bp for histone ChIP and RNA-seq.
This preserves native 128-bp ChIP targets and keeps one unambiguous resolution
for every family.

For 1-bp count-like tracks, 128-bp targets are sums, not means. If a source has
a spatial validity mask, aggregation stores both the sum and the number of
valid bases; a bin with no valid bases is invalid. Track-level masks remain
broadcastable and are not expanded into redundant per-base arrays.

The track manifest contains only real human outputs presented to a model, but
also records each track's source/padded channel index. It includes stable track
ID, family-local index, source index, name, assay, biosample, ontology, strand,
native resolution, stored resolution, transform, dtype, mask semantics, and
metric. `SPLICE_SITE_POSITIONS` is an auxiliary coordinate bundle, not a
twelfth predicted family.

## Split policy

The dataset-capacity priority is:

1. retain every real human track in all 11 output families;
2. keep the final release below the 100-GiB hard limit;
3. allocate most example capacity to training;
4. retain a smaller but genomically distributed test subset;
5. retain a minimal validation subset for model selection.

### Test

- The complete official human `FOLD_0` fold1 source-record manifest is pinned
  and retained as metadata.
- The materialized test set is a fixed 1,000-example subset selected from that
  manifest without inspecting targets or model scores.
- Test records are never reconstructed on a new lattice. Selection is
  deterministic and stratified by chromosome and source shard.
- Bundle absence is represented by a missing-target mask rather than removing
  the example.
- All 11 output families and all real human tracks are retained for every
  selected test example.
- Both the full official manifest and selected test manifest are shipped; the
  selected IDs are frozen as part of the protocol.

### Validation

- The complete official human `FOLD_0` fold0 manifest is pinned as metadata.
- The materialized validation set is a deterministic 200-example subset,
  stratified by chromosome and source shard without using target values.
- Validation is not used to fit parameters and is never merged into training.
- All output families and tracks are retained; validation is made smaller by
  reducing examples, never by reducing the output space.

### Training

- Candidate training examples come only from the remaining official training
  folds (fold2 through fold7 under `FOLD_0`).
- Overlap filtering happens before downsampling.
- Full and selected test/validation manifests are immutable; an offending
  training example is removed instead.
- Downsampling is deterministic, label-independent, and reproducible from a
  published seed and stable example ID.
- Both the complete eligible-training manifest and the selected affordable
  subset are saved, making the exact training subset auditable.

The frozen materialized training budget is 3,200 examples after overlap
filtering. A 1 Mb, 11-output example has substantially higher compute and I/O
cost than a short-context accessibility-only example.

Downsampling is stratified by source fold, chromosome, and source shard. It
must not inspect target magnitude, biosample activity, or any model score. If a
hard byte budget is used, its value and the resulting count are frozen in the
dataset manifest.

### Frozen count and storage budget

The complete source index and release validator are the authorities for these
counts.

| Split | Official BED intervals | Materialized protocol |
| --- | ---: | ---: |
| train, fold2--fold7 | 41,694 | 3,200 selected after filtering |
| validation, fold0 | 7,090 | 200 selected; full manifest retained |
| test, fold1 | 6,713 | 1,000 selected; full manifest retained |

An interval sweep with 1,048,576-bp contexts gives these training survivor
counts:

| Exclusion rule | Removed | Surviving |
| --- | ---: | ---: |
| train target versus valid/test target | 0 | 41,694 |
| train context versus test target | 330 | 41,364 |
| train context versus valid/test target | 674 | 41,020 |
| train context versus test context | 691 | 41,003 |
| train context versus valid/test contexts | 1,344 | 40,350 |

The frozen exclusion rule is `train context versus valid/test target`, leaving
41,020 eligible intervals before deliberate downsampling. Under that rule:

- selecting 3,200 retains 7.80% of eligible training intervals;
- selecting 200 validation examples retains about 2.82% of fold0 intervals;
- selecting 1,000 test examples retains about 14.90% of fold1 BED intervals
  (16.41% of the 6,094 source-backed records).

The release contains 4,400 examples: 3,200 train, 200 valid,
and 1,000 test. At 1 Mb input, the training split processes about 3.36 billion
input bases per epoch while providing long context and all output families.

The current public compressed source inventory for all bundles, including the
auxiliary splice-site-position bundle, is approximately:

| Split | Source TFRecords |
| --- | ---: |
| train | 8.23 TiB |
| validation | 98.0 GiB |
| test | 383.6 GiB |
| total | 8.70 TiB |

Raw train TFRecords can be streamed and discarded after selected examples are
materialized; they need not all remain on disk. A straightforward float32
representation would require roughly 45--70 MiB per example and cannot meet
the release limit. The protocol therefore requires compact storage: float16/bfloat16
where a fidelity test passes, triangular contact-map storage, sparse splicing
events, broadcast masks, and Zarr compression.

| Materialized split | Examples | Measured targets |
| --- | ---: | ---: |
| train | 3,200 | 64.55 GiB |
| validation | 200 | 4.04 GiB |
| test | 1,000 | 20.26 GiB |
| **total** | **4,400** | **88.86 GiB** |

The complete installed release is 92.03 GiB including the 3.09-GiB pinned
reference and manifests. The validator's hard limit is 100 GiB. Raw upstream
TFRecords are not part of the release.

## Leakage and overlap rules

Reference DNA is public and identical across splits, so shared sequence context
alone is not target leakage. Requiring all 1 Mb context windows to be mutually
disjoint can remove a large fraction of training data near fold boundaries and
is not necessary for preserving held-out labels.

The required exclusion rule is:

```text
drop train example if
  train_target intersects test_target or validation_target
  OR any target label assigned to the train example intersects a held-out
     target interval
```

Overlap filtering uses the complete official fold0/fold1 manifests, not only
the 200/1,000 materialized subsets. Compact evaluation therefore does not relax
the original held-out genomic boundary for training selection.

The audit also computes, without hiding it:

```text
train_context intersects test_target
train_context intersects test_context
validation_context intersects test_context
```

As an additional conservative guard, the protocol may drop a training example when its
1 Mb input context intersects a test target interval. This guard must be frozen
before release and its removed/surviving counts reported. We do not recommend
requiring `train_context` to be disjoint from the full `test_context`: that
tests shared public reference sequence rather than label leakage and may
unnecessarily destroy the affordable training set.

The frozen protocol does not require full-context isolation. The audit reports
validation-versus-test context overlap without creating another split profile.

## Storage and streaming

The official source is already sharded by bundle. The protocol preserves that property
and does not construct a monolithic example-by-position-by-track array.

Logical layout (the top-level directory name is a versioned wire-format
detail):

```text
dataset/
  dataset_manifest.json
  target_schema.json
  tracks.parquet
  splits/
    train_eligible.parquet
    train_selected.parquet
    valid_fold0_full.parquet
    valid_fold0_selected.parquet
    test_fold1_full.parquet
    test_fold1_selected.parquet
  shards/
    <split>/<source-shard>/
      dense_128.zarr
      splice_sites/
      splice_site_usage/
      splice_junctions/
      contact_maps.zarr
```

Dense arrays are chunked along examples, positions, and modest track blocks.
Splicing targets use coordinate/value or pairwise event storage. Contact maps
may store only the valid triangle when the decoder reconstructs the exact
symmetric mask. DNA is reconstructed from the pinned human FASTA and stored
coordinates rather than duplicated for every family.

Every shard records source object names, generations/checksums, processed-file
checksums, example IDs, shape, dtype, transform version, and track-manifest
hash. Interrupted materialization resumes only from verified shard boundaries.

## Evaluation protocol

The verifier requests and grades one prediction shard at a time. A submission
does not write one full-test `predictions.npy`.

For dense families, the grader maintains float64 sufficient statistics per
track: count, sum of target, sum of prediction, squared sums, cross-product,
and squared error. Masks are applied before every update. This produces:

- per-track Pearson and error diagnostics;
- the mean of valid per-track scores for each assay family;
- optional global family Pearson as a secondary diagnostic.

For task-specific families, splice usage is gathered at hidden stored events,
Junctions use the 8,192 frozen legacy trusted coordinate queries plus at most
one deterministic supervised query per available track/example (8,926 maximum)
and model-predicted
endpoints, and contact maps use the stored valid upper triangle. Continuous targets
are compared in `asinh` space. The grader reports per-track Pearson, the mean
for each of the eleven families, and an equal mean across families.

## AlphaGenome reference evaluation

The reference run uses official AlphaGenome `FOLD_0` weights, not an
`all_folds` checkpoint. A reference adapter must:

1. read the same selected test manifest and 1 Mb coordinates;
2. request the same human tracks in manifest order;
3. crop to the stored target interval;
4. apply the same 128-bp pooling or event gathering;
5. apply the same masks;
6. submit shards to the same grader.

This gives an apples-to-apples comparison on the compact protocol. It does
not make these scores numerically interchangeable with AlphaGenome paper metrics
that use different resolution, target spans, examples, or aggregation.

## Release verification

The source audit aligned all twelve required bundles by coordinate and DNA
hash. Indexing found 41,020 eligible train, 1,617 source-backed validation, and
6,094 source-backed test records before frozen selection. The final
3,200/200/1,000 materialization contains every one of the 5,930 tracks and
passed the complete structural, identity, overlap, and 100-GiB checks. `DATASET_STATS.md`
records the measured split sizes; `PIPELINE.md` records the operator and
integrity details.
