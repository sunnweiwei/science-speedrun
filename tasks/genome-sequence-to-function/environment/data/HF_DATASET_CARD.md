---
license: cc-by-4.0
pretty_name: AI4Sci genome-sequence-to-function (AlphaGenome-scale) release v4
tags:
- genomics
- functional-genomics
- alphagenome
- variant-effect-prediction
---

# genome-sequence-to-function: prepared release v4

This is the data for the `genome-sequence-to-function` task of
[T0-RSI/ai4sci-tasks](https://github.com/T0-RSI/ai4sci-tasks): one model that maps 1,048,576 bp
of human DNA to 5,930 functional-genomics tracks in eleven families, and predicts variant effects.
It is built from AlphaGenome's public training data (`gs://alphagenome-datasets/v1/train`,
`FOLD_0`) by the task's `environment/data/materialize.sh`. Use that script to install it; it
pins this repository's revision and verifies every archive.

| part | content | stored |
|---|---|---:|
| train targets | all 41,020 eligible intervals of folds 2–7, 8-bit codes of `asinh(x)` with per-track ranges | 308 GiB |
| validation / test targets | 200 (fold 0) / 1,000 (fold 1) intervals, full precision | 4 / 21 GiB |
| support | manifests, labels, junction scoring queries, track metadata, GRCh38.p13 FASTA, variant evaluations (test queries and labels, labelled development split) | 4 GiB |

**Layout.**
- `archives/support.tar` holds every file except the targets.
- `archives/train/*.tar` holds one tar per Zarr train shard.
- `archives/{valid,test}/*.zarr.zip` are Zarr ZipStores.
- `release-manifest.json` gives the size and SHA-256 of each of these.

Extracting everything into one directory recreates the release that the task's validator checks.

**Test data.** The test targets and the test variant labels are included because the upstream
data are public. Harbor mounts them only into the task's grader, never into the agent's
workspace.

**Sources and licenses.**
- AlphaGenome training data and evaluation tables: Avsec et al., Nature 649 (2026), CC BY 4.0.
- TraitGym (Benegas et al. 2025): MIT.
- GENCODE v46.
- GRCh38.p13 reference genome (GENCODE).

Cite the original works when using this release.
