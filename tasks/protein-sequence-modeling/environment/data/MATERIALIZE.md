# Materializing the PEER data (controller and maintainers)

This file is not copied into the agent's workspace. Set `TASK_DATA_DIR` and `TASK_PRIVATE_DIR` to
the materialized root (`/path/to/peer-data`):

- the development compose mounts the entries of `public/development/` under `/workspace/data`;
- the verifier compose mounts `public/verifier/` (inputs) and `private/` (labels, grader only).

## Materialization

```bash
pip install lmdb numpy
conda install -c bioconda mmseqs2          # required for decontamination

bash materialize.sh /path/to/peer-data     # PEER tasks: download, convert, split

curl -o /path/to/peer-data/cache/uniref50.fasta.gz \
  https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz
curl -o /path/to/peer-data/cache/uniref50.release_note \
  https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.release_note
python3 build_pretrain_subset.py /path/to/peer-data --all \
  --release-note /path/to/peer-data/cache/uniref50.release_note \
  --workdir /fast/local/scratch --threads 128
```

Both are **operator-run**, never executed automatically by Harbor, and safe to
rerun. They produce:

```text
/path/to/peer-data/
  cache/                       # downloads + extracted archives (deletable)
  public/development/          # each entry mounted read-only under /workspace/data in the dev container
    manifest.json  stats.json
    fit/<task>.jsonl  tune/<task>.jsonl
    pretrain/uniref50_subset.fasta  pretrain/manifest.json
  public/verifier/             # mounted read-only at /benchmark for the verifier
    manifest.json  <task>.jsonl        # inputs only, no labels
  private/                     # mounted only into the trusted grader service
    manifest.json  <task>.jsonl        # {"example_id","label"} rows
```

`pretrain/manifest.json` records the UniRef50 release note, the source file's
size and SHA-256, whether it matched the pinned release, the sampling seed and
scan counts, how many sequences the decontamination removed, and the resulting
length distribution. A build whose `pinned_release` is null came from a
drifted source and must not be compared with the shipped references.

Split mapping: PEER `train` → `fit`, `valid` → `tune`, and the PEER-standard
test split → the held-out eval view (`test` for most tasks,
`test_fold_holdout` for `fold`, `cb513` for `secondary_structure`). The
remaining upstream splits (fold family/superfamily holdouts, SSP
casp12/ts115, PPI cross-species tests) are not materialized. Exact expected
row counts are asserted in `materialize.py`; a count or checksum mismatch
fails loudly.

`stats.json` records fit-split label means/stds (regression) and class counts
(classification, SSP) — used by the starter and by the `ppi_affinity` reward
normalization (the grader reads the std from the private manifest).

## Wiring the environments

Set the environment variable consumed by the compose files:

```bash
export TASK_DATA_DIR=/path/to/peer-data       # development entries (public/development/*)
export TASK_PRIVATE_DIR=/path/to/peer-data    # verifier inputs (public/verifier) and grader labels (private)
```

The development container never sees `public/verifier` or `private`. The
verifier inference service mounts inputs only; labels are mounted solely into
the separate grader service (see [../../tests/README.md](../../tests/README.md)
for the actual isolation boundary and its limits).
