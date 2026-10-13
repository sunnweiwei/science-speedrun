# Compact human AlphaGenome pipeline

This is the operator guide for the only released protocol.

## Execution environment

Networked data operations and tests are run on the GCP Compute Engine instance
`download` in `us-central1-a`. Do not download AlphaGenome TFRecords through a
developer workstation. Git/source synchronization is small control-plane
traffic; all GCS payload traffic must originate on the remote worker.

## Implemented stages

The `agbench` CLI now provides:

- `bootstrap`: writes the complete coordinate manifests, provisional
  deterministic selections, and the 5,930-track human manifest;
- `inspect`: inventories all twelve source bundles and optionally checks the
  first record's fields, shapes, coordinates, and DNA hash;
- `index`: streams only `SPLICE_SITE_POSITIONS` to replace provisional BED
  selections with authoritative source-backed record selections;
- `smoke`: transforms one aligned real train or valid record across every family;
- `materialize`: writes resumable source-shard Zarr parts for a selected
  split;
- `validate`: validates family shapes, track counts, selected IDs, sparse
  structures, resume identity, completeness, and the 100-GiB target limit.

The remote wrapper is:

```bash
bash environment/data/materialize.sh bootstrap /path/to/release
bash environment/data/materialize.sh reference /path/to/release
bash environment/data/materialize.sh probe /path/to/release
bash environment/data/materialize.sh smoke /path/to/release
bash environment/data/materialize.sh index /path/to/release
bash environment/data/materialize.sh full /path/to/release
bash environment/data/materialize.sh validate /path/to/release
```

On a shared VM, `ALPHAGENOME_MATERIALIZER_MEMORY`,
`ALPHAGENOME_MATERIALIZER_MEMORY_SWAP`, and
`ALPHAGENOME_MATERIALIZER_CPUS` apply Docker cgroup limits to each stage.
`ALPHAGENOME_MATERIALIZER_SKIP_BUILD=1` reuses an already verified image.
An OOM-killed stage exits without a completed part; rerunning `full` rebuilds
only its partial shard and keeps completed parts.

`ALPHAGENOME_MATERIALIZER_WORKERS` controls independent source-shard workers
within each split (default `1`). For example, on a dedicated 16-vCPU,
64-GiB-RAM VM, set it to `4` and run `full`; no GPU is used by materialization.
Each worker handles one complete shard in a fresh spawned process so TensorFlow
memory is released between shards. Completed Zarr parts remain atomic and
resumable, and the combined manifest is sorted by source shard rather than
completion order. The Docker `MEMORY` limit applies to **all** workers together,
not to each worker. The tested serial container completed real train shards
under an 8-GiB hard cap; budget roughly 10–12 GiB RAM per parallel worker,
plus OS/file-cache headroom, until multi-worker peak usage is measured.
Start with 2 workers on 32 GiB or 4 workers on 64 GiB rather than assuming
linear CPU scaling; GCS input and disk output may become bottlenecks.

`full` is locked until both train and valid real all-family smoke receipts and
the source-index receipt exist. It also fetches and verifies the pinned
GRCh38.p13 FASTA before target materialization.

## Representations

- ATAC, DNase, PRO-cap, CAGE, RNA-seq: sum-pooled from 1 bp to 128 bp.
- TF and histone ChIP-seq: native 128-bp targets.
- Dense masks remain track-level when the source mask is broadcastable; any
  non-finite value promotes the mask to a spatial-track mask.
- Splice sites: four real one-base channels.
- Splice-site usage: values and validity masks gathered only at valid positions
  from the official `[4, 512]` splice-position tensor.
- Splice junctions: sparse COO values plus the official site positions and
  explicit `[512, 512, 734]` dense-shape metadata. Random query pairs with
  an invalid site axis are excluded from training loss and scoring.
- Contact maps: float16 symmetric upper triangle plus a validity mask at native
  2,048-bp resolution; the smoke receipt records cast error.
- DNA is represented by pinned reference coordinates and source DNA hashes,
  not duplicated in each target shard.

Official TFRecord intervals may include symmetric context outside the frozen
model input. Indexing keeps an exact 1,048,576-bp centered input; for a
1,052,672-bp source interval this deterministically crops 2,048 bp from each
edge. Both original source coordinates and crop offsets remain in the manifests.
The one-dimensional targets are independently cropped to the centered
196,608-bp target span; train source signals are padded whereas valid/test
signals are often already target-sized. Train contact maps are cropped from
514 to 512 bins to match the model input. Splice junction axes remain 512
sites, with site positions converted to model-input-relative coordinates.

All family-to-tensor channel mappings come from `tracks.parquet`. Padding
channels are excluded while source/tensor indices remain explicit.

## Remote verification completed

The real `FOLD_0/HOMO_SAPIENS/VALID` probe confirmed aligned coordinates and
DNA hashes across all twelve bundles. Observed task-specific source shapes are:

```text
splice_sites          [196608, 5]       -> 4 real tracks
splice_site_usage     [196608, 734]
splice_junctions      [512, 512, 734]
splice_site_positions [4, 512]
contact_maps          [512, 512, 28]
```

The track manifest contains exactly 5,930 real outputs. The complete 4,400
examples passed the release validator with 88.86 GiB of targets and 92.03 GiB
installed including the verified 3.09-GiB reference.

A real 1-Mb example also passed an eleven-family forward, loss, and backward
step with every family loss finite. Its contact mask retained 2,382,702 of
3,677,184 upper-triangle values; this check is why contact validity is an
explicit stored target rather than an implicit numerical convention.

Source indexing has confirmed:

| Split | BED candidates | Source-backed records | Selected |
| --- | ---: | ---: | ---: |
| train/fold2--fold7 | 41,694 | 41,020 after 674 policy exclusions | 3,200 |
| validation/fold0 | 7,090 | 1,617 | 200 |
| test/fold1 | 6,713 | 6,094 | 1,000 |

All 325 source shards in the three splits were scanned without an unmatched
source record. The full official BED manifests remain packaged alongside the
fixed source-backed selections.

## Resume and integrity behavior

Each completed Zarr part pins all twelve source object identities and embeds
its materialized manifest rows. Resume accepts an existing part only when its
schema and selected example IDs exactly match the frozen selection. Partial
parts are removed and rebuilt from the shard boundary.

The source object generation, size, MD5, and CRC32C are checked against the
inventory both before and after every newly scanned or materialized TFRecord
object. Per-shard index checkpoints pin the same identity, so a resumed scan
cannot silently combine two upstream object versions.

Record-index filtering occurs before TensorProto parsing. This is essential
for junction records: unselected `[512,512,734]` tensors are streamed through
gzip but never materialized as dense NumPy arrays. Selected tensors are read
from their serialized binary content without an additional full TensorFlow
tensor copy; padded train targets are cropped before NumPy allocation.
