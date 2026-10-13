---
license: mit
language:
- en
pretty_name: AI4Sci MassSpecGym Retrieval — Materialized Development and Verifier Inputs
task_categories:
- other
tags:
- chemistry
- mass-spectrometry
- molecular-retrieval
- ai4sci
- benchmark
size_categories:
- 1M<n<10M
---

# AI4Sci MassSpecGym retrieval: prepared public inputs

This dataset distributes the already-materialized data for the
[AI4Sci molecular retrieval task](https://github.com/T0-RSI/ai4sci-tasks/tree/main/massspecgym-retrieval).
It preserves the official MassSpecGym v1.5 structure-disjoint split and task
candidate IDs/order. Download the pinned release using the task's materializer;
no raw-source sanitation or MAGMa feature preparation is required.

![Scientific problem, MS/MS input, molecular ranking, and scientific value](figures/science-introduction.png)

This real validation example shows a recorded GLACIER retrieval error (target
rank 22 among 256 candidates), not an idealized successful model prediction.
The figure and its source provenance are supplied with this release.

## Contents

- `development/train.jsonl`: 194,119 experimental training spectra.
- `development/validation.jsonl`: 19,429 validation spectra with labels.
- `development/validation_candidate_pools.jsonl`: mass-filtered candidate pools.
- `development/glacier/`: prepared train/validation MAGMa supervision,
  99,119 / 9,720 annotated spectra, including `targets.sqlite`.
- `development/synthetic_augmentation/`: 2,333,485 held-out-clean ICEBERG
  simulations in a lossless SQLite database and labels/contamination audits.
- `verifier/`: 17,556 test queries and candidate structures, with no test
  answer IDs or target structures attached to the query records.
- `release-manifest.json`: exact payload paths, byte counts and SHA-256 hashes.

The 17 payload files total **38,019,369,914 bytes**. The core source contained
231,104 spectra, with 22,746 / 3,185 / 2,998 unique 2D structures in the
train / validation / test folds. A maximum of 256 candidate molecules is
ranked per query. Candidate SMILES are legitimate model inputs, and their
order is deterministic and answer-independent.

Grader answers are distributed separately in the private operator dataset
`Corning/ai4sci-massspecgym-retrieval-evaluation`. Development mounts only
`development/`; inference mounts only `verifier/`; only the trusted grader
receives the labels. MassSpecGym and its historical labels are public upstream,
so this separation is evaluation hygiene, not a secret-benchmark claim.

## Provenance and license

The MIT-licensed experimental source is
[`roman-bushuiev/MassSpecGym`](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym)
at full revision `d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a`:
`data/MassSpecGym1.5.tsv` and
`data/molecules/MassSpecGym1.5_retrieval_candidates_mass.json`.

The MIT-licensed synthetic source is [Zenodo 20671842](https://zenodo.org/records/20671842),
`msg_aug_filt.tar.gz`, MD5 `32d8813d84d06364fc318e81f9cb4416`.
Task sanitation removed 15 validation/test-structure overlaps and 10 invalid
SMILES rows; the published 2,333,485-row corpus has zero held-out structure
overlap. MAGMa targets derive only from official train/validation data using
the pinned MIT-licensed `coleygroup/ms-pred` source.

Retain the accompanying `UPSTREAM_LICENSES.md` and MIT attribution when copying
or redistributing these views. Cite **MassSpecGym: A benchmark for the discovery
and identification of molecules**, NeurIPS Datasets and Benchmarks 2024, and
the ICEBERG / MIST / GLACIER sources when using their derived resources.
No model weights, released FRIGID checkpoint, private workspace files, or
credentials are included. The reference score does not grant permission to use
pretrained weights: this task's agent training policy remains from scratch.

## Benchmark contract

Input: experimental MS/MS peak lists, precursor/adduct/instrument/energy
metadata, and a mass-filtered candidate structure set. Output: one complete
candidate permutation per query. Primary scientific metric: full-denominator
Hit@1; secondary metrics: Hit@5, Hit@20, and MCES@1.

The fixed score anchors are 0.04004329004329004 (repaired Fourier-DeepSets)
and 0.4939052175894281 (official GLACIER no-CF + clean MIST reference).
The benchmark score is `clip((Hit@1 - lower)/(upper - lower), 0, 1)`.
Raw metrics and unclipped scores remain reported. The upper anchor is an
empirical reference, not a theoretical optimum. See the GitHub task for
immutable revision pins, full provenance, and submission/evaluation rules.
