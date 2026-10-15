# PEER data

Schema `peer-protein-input/v1`. Row and prediction formats are specified in
`/workspace/contract.md`. In this workspace, `/workspace/data` holds this file and four read-only
mounts:

```text
/workspace/data/manifest.json, stats.json       split manifest; fit-split label statistics
/workspace/data/fit/<task>.jsonl                 labeled training rows
/workspace/data/tune/<task>.jsonl                labeled rows for model selection
/workspace/data/pretrain/uniref50_subset.fasta   unlabeled UniRef50, decontaminated against the held-out sequences
/workspace/data/pretrain/manifest.json           how the corpus was built
```

The held-out test inputs and labels are not in this workspace.

## Sources

All eleven datasets are the public distributions used by the
[PEER benchmark](https://arxiv.org/abs/2206.02096) via
[torchdrug](https://github.com/DeepGraphLearning/torchdrug); each archive URL
and md5 is pinned in the task's `materialize.py`. Four archives
(`fluorescence`, `stability`, `secondary_structure`, `remote_homology`) are the
original [TAPE](https://arxiv.org/abs/1906.08230) LMDB releases on
`songlabdata` S3; the rest are PEER re-releases on
`miladeepgraphlearningproteindata` S3. Total download is roughly 1–2 GB;
allow ~5 GB of disk for cache plus converted views. Data is **not**
redistributed in this repository. Check the upstream licenses of TAPE and
PEER before redistribution; original per-dataset credits (GFP fitness,
stability, DeepSol, DeepLoc, SCOPe remote homology, NetSurfP/CB513, STRING
PPI, SKEMPI) are listed in the PEER paper.

The archives contain pickled LMDB records; `materialize.py` unpickles them, so
run it only against the pinned, checksum-verified downloads.

## Split counts, measured

torchdrug's dataset docstrings are the usual source for PEER split sizes, and
they **disagree with the archives torchdrug itself distributes for 16 of the 33
splits**. All eleven md5 checksums match, and `load_lmdbs` applies no
filtering, so the measured numbers below are what anyone loading PEER today
actually gets. Two distinct causes:

- **The three PPI docstrings count unique proteins, not labeled pairs.** Every
  one of the nine PPI split numbers matches exactly once you count distinct
  sequences instead of rows: `human_ppi` train is 6,844 unique proteins forming
  **35,669** pairs, `yeast_ppi` train is 1,668 proteins forming **4,945** pairs,
  `ppi_affinity` train is 2,127 proteins forming **2,421** pairs. Anyone citing
  "6,844 training examples" for human_ppi is off by a factor of five.
- **`stability`, `subcellular_localization` and `binary_localization` were
  re-split upstream** after the docstrings were written; the totals differ too,
  so it is not a boundary shift alone.

| Task | fit | tune | eval | eval split |
| --- | --- | --- | --- | --- |
| fluorescence | 21,446 | 5,362 | 27,217 | test |
| stability | 53,614 | 2,512 | 12,851 | test |
| beta_lactamase | 4,158 | 520 | 520 | test |
| solubility | 62,478 | 6,942 | 1,999 | test |
| subcellular_localization | 8,420 | 2,811 | 2,773 | test |
| binary_localization | 5,184 | 1,729 | 1,749 | test |
| fold | 12,312 | 736 | 718 | test_fold_holdout |
| secondary_structure | 8,678 | 2,170 | 513 | cb513 |
| yeast_ppi | 4,945 | 95 | 394 | test |
| human_ppi | 35,669 | 315 | 237 | test |
| ppi_affinity | 2,421 | 203 | 326 | test |
| **total examples** | **219,325** | **23,395** | **49,297** | |
| **total sequences** | **262,360** | **24,008** | **50,254** | |

Label cardinality was verified against the data as well: `fold` really does
span 1,195 classes (labels 0–1194, all present) and
`subcellular_localization` 10. Across all splits the labeled corpus is about
**98.2M residues**.

## Pretraining corpus

The labeled pool is small next to what a protein language model needs, so the
task also ships **unlabeled UniRef50**, built by the task's
`build_pretrain_subset.py`. Source:
`https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz`
(CC BY 4.0). That URL always serves the *current* UniRef release, which UniProt
replaces every eight weeks, so the builder pins **release 2026_03
(02-Sep-2026): 8,780,552,383 bytes, SHA-256
`95bdf597d2f892295b5aa2925f5da593e1f2e1fcd7a0a4f5e66b9f1c2e83290d`,
38,840,027 clusters, 37,384,515 within 30–1024 residues** and refuses to build
from anything else unless `--allow-drift` is passed. Once 2026_03 leaves the
current slot it is only available inside
`previous_releases/release-2026_03/uniref/uniref2026_03.tar.gz`. The shipped
reference numbers were first produced on release 2026_02 (8,770,260,598 bytes,
38,794,121 clusters, 12,171,016,896 residues, mean length 313.7; 37,340,018
clusters within 30–1024 residues averaging 265.4) and reproduced on 2026_03
(5M-sequence subset: 4,993,922 sequences after decontamination, 1,324,451,152
residues, mean length 265.2).

Two properties the builder exists to enforce, both easy to get wrong by hand:

- **`uniref50.fasta.gz` is sorted by sequence length, descending.** Its first
  record is a 46,734-residue transcriptional regulator, and the first 12,513
  entries have a median length of 7,969. Taking the head of the file yields a
  corpus with nothing in common with the proteins PEER asks about, so the
  builder reservoir-samples over the full stream with a fixed seed, keeping
  only 30–1024 residues (1024 being ESM-2's crop length).
- **PEER's held-out proteins are natural sequences that live in UniRef50.**
  Without removal, pretraining leaks the test set. The builder drops every
  candidate within 50% identity of any of the 50,254 eval sequences, using the
  same MMseqs2 settings ESM-2 used to decontaminate its own validation split
  (`--min-seq-id 0.5 --alignment-mode 3 -c 0.8`). MMseqs2 is required; the
  builder aborts rather than silently skipping this step. Verified by positive
  control: planting two verbatim eval sequences into a corpus, both were
  removed.

The default (`--all`) is **every in-range cluster: 37,340,018 sequences**, from
which decontamination removes 45,979, shipping **37,294,039 sequences ≈ 9.89B
residues**. Not a subsample — there is no reason to withhold
data the candidate could otherwise sample from, and unrepeated data is strictly
better than repeated data.

Sizing this was where the reasoning had to be corrected. The first instinct is
to ask whether a candidate can finish an epoch, and to subsample until the
answer is yes. That is the wrong question: pretraining samples minibatches from
a corpus and never needs to complete a pass — ESM-2 itself repeated its ~65M
sequences roughly 58 times over 1e12 tokens. Seeing 25% of a large corpus once
beats seeing 100% of a small one four times.

What the budget actually allows, measured on one A100 with length-bucketed
token-budget batches:

| model | throughput | one epoch, 1 GPU |
| --- | --- | --- |
| `esm2_t6_8M` | 495K tok/s | 5.6 h |
| `esm2_t12_35M` | 198K tok/s | 14.0 h |
| `esm2_t30_150M` | 68K tok/s | 41.0 h |

An 8M-scale model therefore covers the whole corpus in under six hours and a
35M-scale one in well under a day, so a research loop can afford repeated full
passes at small scale or partial passes at larger scale — choosing between them
is an experiment, not a constraint imposed by data volume. `--size N` still
takes a random sample instead, and `--size 100000` is the right smoke-test
setting.

**Batch by tokens, not by sequences.** Over this length distribution (mean 265,
p99 1659) fixed 64-sequence batches pad to the batch maximum and waste **3.4× of
the compute** — real tokens are only 29.3% of what is processed. Sorting by
length before forming token-budget batches removes essentially all of it
(99.9% efficiency) and measures 4× faster end to end. ESM-2 batches this way
too ("2 million tokens as batch size"); a candidate that uses fixed-count
batches is giving up most of its budget.

### Measured builds

Sampled runs against the current release, for calibration:

| | `--size 200000` | `--size 5000000` |
| --- | --- | --- |
| Wall clock | ~14 min, 128 threads | 1h 23m, 64 threads |
| Sequences kept | 199,745 | 4,993,902 |
| Residues | 52,998,944 | 1,324,789,425 |
| FASTA | 56 MB | 1.4 GB |
| Contamination removed | 255 (0.128%) | 6,098 (0.122%) |
| Length | mean 265.3, median 203 | mean 265.3, median 203 |

Two things these confirm. Sampled mean length (265.3) matches the
whole-database figure for the same range (265.4), so the reservoir sampling is
unbiased. And the contamination rate is stable across a 25× size change
(0.128% → 0.122%), so it is a property of the corpus rather than sampling
noise. Small as it is, it matters: those sequences are near-duplicates of
held-out test proteins, and a `fold` or `subcellular_localization` model that
memorized them would be scored on recall rather than generalization.

Cost scales better than linearly — 25× the queries took only 6× the wall clock
(on half the threads), because MMseqs amortizes its target index, which is what
makes the `--all` build affordable. The scan is a fixed ~7 minutes regardless
of size, since the whole 8.2 GiB file is read either way. `--threads` is passed
through, and `--workdir` should point at fast node-local storage for `--all`.
The `-s 7` sensitivity is
ESM-2's setting, kept deliberately: lowering it would be faster but would let
more remote homologs of the test set survive.

