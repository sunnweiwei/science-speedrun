# Dataset statistics

This task has one frozen release profile. The complete 4,400-example
materialization passed the structural and 100-GiB release validator.

| Split | FOLD_0 source folds | Official BED rows | Source-backed / eligible | Frozen selected count |
| --- | --- | ---: | ---: | ---: |
| train | fold2--fold7 | 41,694 | 41,020 after 674 overlap exclusions | 3,200 |
| validation | fold0 | 7,090 | 1,617 | 200 |
| test | fold1 | 6,713 | 6,094 | 1,000 |
| **Total** |  | **55,497** |  | **4,400** |

Selections are deterministic and target-independent. Full official coordinate
and source-backed manifests are retained. Held-out manifests are fixed first;
training contexts that intersect a validation/test target are removed before
training downsampling. Validation and test are reduced by example count, never
by dropping output tracks.

## Shapes and tracks

| Family | Tracks | Stored target |
| --- | ---: | --- |
| ATAC | 167 | 1,536 x tracks, 128-bp sum bins |
| DNase | 305 | 1,536 x tracks, 128-bp sum bins |
| PRO-cap | 12 | 1,536 x tracks, 128-bp sum bins |
| CAGE | 546 | 1,536 x tracks, 128-bp sum bins |
| RNA-seq | 667 | 1,536 x tracks, 128-bp sum bins |
| TF ChIP-seq | 1,617 | 1,536 x tracks, native 128 bp |
| histone ChIP-seq | 1,116 | 1,536 x tracks, native 128 bp |
| splice sites | 4 | 196,608 x 4, 1 bp |
| splice-site usage | 734 | valid-site events x tracks |
| splice junctions | 734 | sparse site-pair events |
| contact maps | 28 | 512-bin symmetric upper triangle, 2,048 bp |
| **Total** | **5,930** | eleven output families |

Every example uses a centered 1,048,576-bp model input. Official source
intervals may be larger; a 1,052,672-bp interval is deterministically cropped
by 2,048 bp per edge. The central one-dimensional target span is 196,608 bp.
DNA is referenced by coordinates and is not copied into each target shard.
Track ordering, source tensor index, assay, biosample,
strand, resolution, transform, mask semantics, and metric live in
`metadata/tracks.parquet`.

The source-index audit found split-specific context lengths:

| Split | Source-backed rows | Official source interval | Symmetric crop to model input |
| --- | ---: | ---: | ---: |
| train | 41,020 | 1,052,672 bp | 2,048 bp per edge |
| validation | 1,617 | 1,048,576 bp | 0 bp |
| test | 6,094 | 1,048,576 bp | 0 bp |

This crop is part of the coordinate contract. Some train source tensors include
the full 1,052,672-bp padded interval; these are cropped to the same centered
196,608-bp target as validation/test before storage. The train contact map is
cropped from 514 to 512 bins on both axes to match the 1-Mb input.

## Storage

Measured release storage is:

| Component | Examples | Bytes | GiB |
| --- | ---: | ---: | ---: |
| train targets | 3,200 | 69,313,134,802 | 64.55 |
| validation targets | 200 | 4,335,822,043 | 4.04 |
| test targets | 1,000 | 21,758,806,216 | 20.26 |
| **all targets** | **4,400** | **95,407,763,061** | **88.86** |
| pinned reference |  | 3,321,608,984 | 3.09 |
| manifests/metadata |  | 91,715,152 | 0.09 |
| **complete installed release** | **4,400** | **98,821,087,197** | **92.03** |

These values come from the complete `PASS` validation receipt with
`require_complete: true`; the validator enforces the 100-GiB installed-release
limit.
