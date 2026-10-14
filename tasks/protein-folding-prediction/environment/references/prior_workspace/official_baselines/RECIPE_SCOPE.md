# Frozen data versus synchronized Recipe

The dataset is an experiment control, not part of an upstream model Recipe.
Every trainable candidate starts from the same manifest and structure-grouped
split:

| Control | Frozen value |
| --- | --- |
| dataset | `FoldingTrain-100K` |
| train examples | 98,977 |
| validation examples | 1,023 |
| train SHA-256 | `bdd13b8cfba6a4c39d0223f4b92a15ed53bbe896f6dfcffe1291e2af887dbb24` |
| validation SHA-256 | `f4c8301d30bef7d5000332e7c0dfd05d6714c995b4889cdf97076cdf83e26e63` |
| structure-group overlap | 0 |

An upstream data source, mixture ratio, curriculum, or sampling weight is never
copied into a baseline. A native adapter may only:

1. convert the frozen records into the upstream batch schema;
2. deterministically remove records the model contract cannot represent;
3. report source hashes, accepted/rejected counts, and rejection reasons.

It may not add a paper-specific dataset, replace an unsupported record, move a
structure across the frozen split, or use validation/evaluation ground truth as
an input feature.

The synchronized **non-data Recipe** consists of the following, when publicly
available: architecture and initialization; loss terms and weights; optimizer
and scheduler; regularization, EMA or LoRA; gradient accumulation/clipping;
precision and distributed behavior; checkpoint/resume semantics; and inference
sampling/recycling settings.

| Candidate | Common-data adapter | Non-data Recipe status |
| --- | --- | --- |
| Protenix 48.68M | Protenix-native PDB projection of the frozen split | Official trainer, losses, permutation logic, optimizer, scheduler, checkpoint and inference paths retained |
| DPLM-2 47.05M | Protein-monomer projection; 42,912 train and 444 validation structures tokenized by the frozen official tokenizer | Official three-stage wrappers and DPLM-2 settings retained; the matching 50M sequence initialization and folding run completed under the controlled pass-1 exposure budget |
| ESMFold2-Fast 49.20M | Protein/all-atom projection plus the exact frozen common ESMC mixed-256 cache | Released Biohub architecture and dataflow retained; missing training orchestration uses the explicitly project-owned `foldingtrain_edm_v1` Recipe |
| OpenDDE structural 48.75M | OpenDDE-native all-atom projection of the same frozen split | Released OpenDDE architecture, structural-token path and denoiser retained; missing training orchestration uses the explicitly project-owned `foldingtrain_edm_v1` Recipe |
| ESMC-6B + Protenix hybrid 48.76M | Same Protenix projection plus frozen common ESMC mixed-256 cache | First-party composition; otherwise uses the same official Protenix non-data Recipe |

There are two independent provenance fields rather than one ambiguous
"official" flag:

| Field | ESMFold2/OpenDDE value |
| --- | --- |
| architecture source | official pinned upstream modules |
| training Recipe source | project-owned reconstruction |

`foldingtrain_edm_v1` trains coordinate diffusion plus an auxiliary distogram
objective with AdamW, warmup/cosine decay and clipping. Confidence calibration
is a later stage and is frozen in the v1 update path. The runners implement
distributed one-pass accounting, atomic checkpoint/resume with per-rank RNG,
EMA, and common-view validation. These settings are hypotheses to evaluate
under the common data/FoldBench protocol, not recovered upstream details.
