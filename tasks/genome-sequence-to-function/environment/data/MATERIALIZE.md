# Materializing the data (controller side)

This file is not copied into the agent's workspace. One materialized directory serves both sides:
the agent's compose file mounts only its development entries, and the verifier reads the test
entries from it.

```bash
export TASK_DATA_DIR=/data/ag TASK_PRIVATE_DIR=/data/ag
```

## Prepared release (default)

The release is on Hugging Face:
[sunweiwei/ai4sci-genome-sequence-to-function-data](https://huggingface.co/datasets/sunweiwei/ai4sci-genome-sequence-to-function-data)
(public, CC BY 4.0). `download-source.json` pins the commit and the SHA-256 of its
`release-manifest.json`, which in turn pins every archive:
- `support.tar`;
- 250 train-shard tars;
- 75 validation and test ZipStores.

```bash
bash environment/data/materialize.sh release /data/ag
```

The installer downloads one archive at a time, checks its size and SHA-256, extracts it and
deletes it. It then runs the full release validator and moves the result into place. Allow about
450 GB. The release was published from the build described below with
`environment/data/publish_hf_release.py`.

## Rebuilding from the sources

```bash
bash environment/data/materialize.sh rebuild /data/ag
```

The release is built from AlphaGenome's public training data:
- TFRecords from `gs://alphagenome-datasets/v1/train`, fold `FOLD_0`;
- the pinned GRCh38.p13 FASTA;
- Borzoi's fold BED.

Every input is pinned in `environment/data/src/agbench/constants.py`. Nothing here needs credentials: the bucket is
public and is read over HTTPS.

## What is built

| part | examples | stored | notes |
|---|---:|---:|---|
| train targets | **41,020** (every eligible interval of folds 2–7) | 330.7 GB (7.69 MiB/example) | dense families and contact maps as 8-bit codes with per-track ranges (`src/agbench/quantize.py` under this directory) |
| validation targets | 200 | 4.4 GB | full precision |
| test targets | 1,000 | 21.8 GB | full precision; verifier only |
| reference FASTA | — | 3.3 GB | pinned GRCh38.p13 |
| variant evaluations | 6,587 test + 2,697 development perturbations | < 0.1 GB | `variant_evals/`; test queries public, test labels verifier-only; development split with labels |

On disk the release occupies about 391 GiB (420 GB): the train targets are 308 GiB of data in
many small Zarr chunk files, which take 362 GiB of filesystem blocks. Allow 450 GB for the release.

Eligibility excludes 674 of 41,694 training intervals whose input context overlaps a validation or
test target. Validation and test are the frozen selections of the earlier release; their files are
unchanged.

### Procedure, step by step

```bash
# 1. image (pinned dependencies; includes ISA-L for gzip)
docker build -f environment/data/materializer.Dockerfile -t agbench .
# 2. metadata, reference, source index, selections (minutes)
for mode in bootstrap reference index; do bash environment/data/materialize.sh $mode /data/ag; done
# 3. targets: stream every train shard (8.2 TiB of gzip TFRecords, never stored)
docker run --rm -v /data/ag:/output agbench materialize --data-dir /output --split train --workers 250
docker run --rm -v /data/ag:/output agbench materialize --data-dir /output --split valid --workers 25
docker run --rm -v /data/ag:/output agbench materialize --data-dir /output --split test --workers 50
# 4. manifests, labels and junction scoring queries, then the full validator
docker run --rm -v /data/ag:/output --entrypoint python agbench -c \
  "from agbench.release import export_release; export_release('/output', ['train','valid','test'], resume=False)"
docker run --rm -v /data/ag:/output agbench validate --data-dir /output --splits train,valid,test
# 5. variant and perturbation evaluations (see below)
bash environment/data/fetch_variant_evals.sh /data/ag-evals
docker run --rm -v /data/ag-evals:/e -v /data/ag:/d:ro --entrypoint python <verifier image> \
  environment/data/build_variant_evals.py --sources /e --release /d --output /d/variant_evals
```

**Cost.** Measured on one 208-core host with 1.8 TB RAM:
- train targets: 2 h 52 min with 250 workers, streaming at a peak of about 1.4 GB/s and holding
  about 610 GB of RAM;
- the export of 41,020 junction scoring parts: about 75 min on one core.

Each source shard is an atomic, resumable part, and a rerun keeps completed parts.

**Reader.** Records stream over HTTPS through ISA-L gzip, and the needed tensors are read straight
from the protobuf wire format as NumPy views (`environment/data/src/agbench/{tfrecords,wire}.py`).
- TensorFlow's GCS reader sustained about 6.5 MB/s per process and stalled under concurrency.
- Copy-based protobuf parsing saturated memory bandwidth at about 0.2 GB/s in aggregate.
- On real records from all twelve bundles, the new path returns byte-identical tensors.
- A dropped connection reopens the stream and skips the records already read.

## Train-target quantization

Dense targets are stored as `uint8` codes of `asinh(x)` between per-track `lo` and `hi`, and
contact maps linearly. The training reader (`AlphaGenomeDataset._decode`) restores them per example.
On real examples the round trip keeps per-track Pearson ≥ 0.9997 in the metric's `asinh` space
(median 0.99999), and the maximum absolute error is half a quantization step.

Validation and test targets are never quantized, so every score is computed against full-precision
targets.

## Variant evaluations

`fetch_variant_evals.sh` downloads:
- AlphaGenome's eleven released evaluation tables
  (`storage.googleapis.com/alphagenome/evals/*_predictions.feather`, CC-BY 4.0), with their
  variants, labels and AlphaGenome's own scores;
- TraitGym (`songlab/TraitGym@1fde195`, MIT), with Borzoi, Enformer, CADD and GPN-MSA scores;
- AlphaGenome's GENCODE v46 annotation.

`build_variant_evals.py` then draws the seeded subsets: the test split
(`variant_evals/public` and `variant_evals/labels`, verifier only) and a disjoint development split
with labels (`variant_evals/valid`, mounted for the agent). The design is in
`docs/EVALUATION_DESIGN.md`. `environment/evaluation/variant_scoring.py`
reproduces all eleven of AlphaGenome's published metrics from its released predictions, to within
0.0004.
