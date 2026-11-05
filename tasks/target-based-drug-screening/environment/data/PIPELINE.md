# Target-based Drug Screening data materializer

This directory is the source-controlled recipe. The materialization target
becomes the complete `data/` tree. Full data are intentionally not committed
and the smoke build is explicitly marked as synthetic.

Use persistent operator-owned directories outside the worktree:

    /large-disk/drugclip/materialized/  # raw, canonical, splits, tests, processed LMDB
    /large-disk/drugclip/pretrained/    # checksum-pinned generic Uni-Mol initialization
    /large-disk/drugclip/development/   # staged agent-visible allowlist
    /large-disk/drugclip/private/       # staged labeled verifier data
    /large-disk/drugclip/research-run-001/
    /large-disk/drugclip/run-records/run-001/

[Prepare a run](MATERIALIZE.md) is the operator entry point. Set
TARGET_BASED_DRUG_SCREENING_DATA_DIR or pass an explicit absolute directory;
the legacy low-level materializer fallback is host-specific. prepare.sh has
a portable task-local default, and DRUGCLIP_DATA_DIR remains a compatibility
fallback. Neither default establishes a reproducibility lock.

Acceptance runs use the recorded immutable HF commit, separate staged-file
fingerprints, exact built-image IDs, and observed runtime versions. See
[the run-record protocol](../../docs/REPRODUCIBILITY.md). Reusing the same
directory or a main revision is not sufficient to establish identical data.

## Official format inspection

The format was inspected at fixed upstream revisions rather than inferred from
field names:

- DrugCLIP code revision `9d0db73915219763b5bcc3b7de86d7c4edb5ed03`.
  `unimol/tasks/drugclip.py::load_dataset` opens `<split>.lmdb` and requests
  `smi`, `pocket`, `atoms`, `coordinates`, `pocket_atoms`,
  `pocket_coordinates`, and `label`. `AffinityDataset` samples one item from
  the ligand conformer list, while pocket coordinates are a single atom × 3
  array. `unimol/data/lmdb_dataset.py::LMDBDataset` reads a single-file LMDB
  (`subdir=False`) whose ASCII numeric keys contain pickled dictionaries.
- `unimol/encode_pockets.py::get_binding_pockets` selects a whole residue when
  the minimum residue-atom/ligand-heavy-atom distance is at most 6 Å;
  `pocket2lmdb` removes hydrogen and emits `pocket`, `pocket_atoms`, and
  `pocket_coordinates`. This pipeline follows that definition.
- `unimol/utils/conf_gen_cal_metrics.py::single_conf_gen` uses RDKit
  `EmbedMultipleConfs`, fixed seeds, hydrogen addition/removal, and MMFF
  optimization. The full official benchmark conformer clustering is designed
  around many bound conformers. Here the fixed limited-compute policy is up to
  10 ETKDGv3 conformers, MMFF94s optimization (UFF fallback), one thread, seed
  derived from protocol seed + complex ID, and 0.5 Å prune RMS.
- `unimol/tasks/drugclip.py::test_dude_target` reads each DUD-E target's
  `mols.lmdb` (`atoms`, `coordinates`, `smi`, `label`) and `pocket.lmdb`.
  `test_pcba_target` uses the analogous LIT-PCBA `mols.lmdb` and plural
  `pockets.lmdb`. The exact released ZIPs also contain the source label files:
  DUD-E `actives_final.ism` / `decoys_final.ism`, and LIT-PCBA `actives.smi` /
  `inactives.smi` (duplicated as positives/negatives). Indices are built from
  these source lists; the released LMDBs remain untouched. All released test
  actives pass strict RDKit sanitization. Four rows among 1,411,214 DUD-E
  decoys contain an author-released over-valent neutral nitrogen; the index
  preserves their unsanitized source graph, canonical SMILES, and InChIKey and
  marks them `unsanitized_source_valence`. This fallback is forbidden for
  actives and therefore cannot weaken active-ligand leakage filtering.
- ProFSA code revision `a056baed91277059acba3836d5fbf3fccfbee605`.
  `conf/dataset/profsa.yaml` and `src/dataset/profsa.py::ProFSADataset` use
  `lig_atoms_real`, `lig_coord_real`, `pocket_atoms`, and
  `pocket_coordinates`, with `smi` and `pocket` metadata. The Hugging Face
  release itself is PDB complexes: chain `R` is the surrounding pocket and
  chain `L` is the protein-fragment pseudo-ligand. The processed view preserves
  native ProFSA keys and includes standard DrugCLIP aliases.
- The BioLiP2 URL now serves Q-BioLiP. Its official protein-small-molecule
  bundle has receptor PDBs named by `Assembly_ID`, ligand structures named by
  `Ligand_file`, FASTA, and `PL_annotation.csv` with
  `Assembly_ID,Ligand_file,Ligand_ID,Binding_residue`. This recipe pins the
  site's static complete redundant PL bundle from before 2026-01-02 and then
  performs its own cleaning and 30% clustering. The official `artlist` is an
  explicit crystallization/buffer exclusion list.

## Dataset scale and compact-protocol differences

This is a DrugCLIP-compatible compact protocol, not a sample-for-sample copy of
the original DrugCLIP training release. Scale reduction is deliberately
concentrated in the synthetic ProFSADB pretraining component; real BioLiP2
complex coverage and the author-released tests are kept as complete as the
quality and leakage policy permits.

| Component | Upstream scale | This protocol | Scale decision |
| --- | ---: | ---: | --- |
| ProFSADB pretraining | over 5 million pseudo-pocket/fragment pairs in the ProFSA release | exactly 500,000 clean pairs | fixed-seed sample after structural validation and test-protein filtering; approximately 9–10% of the upstream pool |
| BioLiP2/Q-BioLiP | 560,724 unique complexes in the pinned annotation | all 560,724 represented in the canonical table | no affordability downsampling; invalid, artifact, leakage, and explicitly documented stalled-tail rows remain indexed with exclusion reasons |
| DUD-E + LIT-PCBA | complete author-released targets and ligand lists | complete author-released targets and ligand lists | no target, active, decoy, or inactive subsampling |

For ProFSADB, the full source pool must still be downloaded and inspected so
test-target leakage can be removed before sampling. Only the selected 500,000
pairs receive derived NPZ arrays and enter the model-ready LMDB. Compared with
the upstream pool this reduces derived storage, epoch length, and training I/O
by roughly one order of magnitude, while leaving the raw source auditable.

BioLiP2 is reduced only by policy filters, not by a target sample count. The
pipeline retains the complete source ID universe in `biolip2.parquet`, then
marks official artifacts, missing/invalid structures, unusable pockets or
conformers, test-protein matches, and exact/similar test-active ligands as
excluded. The pinned release additionally records 11 manual stalled-tail
exclusions (nine `kmer`, two `HEM`), or about 0.002% of the source IDs. Clean
complexes are split by 30%-identity protein clusters into approximately 90%
train and 10% development data. Final clean train/dev counts are properties of
the completed materialization and are reported in `DATASET_REPORT.md`; they
must not be described as fixed before filtering finishes.

The complete DUD-E and LIT-PCBA releases remain the evaluation population.
Their original structures and LMDB files are preserved, while the pipeline
builds a Parquet identity/label index for leakage checks and streaming access.
This retains the native per-target class imbalance and prevents a smaller test
set from making screening metrics artificially easier.

Representation is compact but follows the model's intended input contract:
the protein input is a local pocket rather than the full protein, defined as
whole residues within 6 Å of any observed ligand heavy atom, with hydrogens
removed. BioLiP2 small molecules have at most 10 deterministic ETKDGv3
conformers with MMFF94s optimization (UFF fallback) and 0.5 Å RMS pruning. Ten
conformers matches the released DrugCLIP data contract, so it is not an extra
conformer-count reduction. ProFSADB retains its released `R`-chain surrounding
pocket and `L`-chain protein-fragment pseudo-ligand coordinates instead of
inventing small-molecule conformers.

Consequently, comparisons should be phrased precisely: this protocol keeps the
released virtual-screening tests intact and uses DrugCLIP-compatible tensors,
but its training corpus and leakage-aware split are newly materialized. It is
appropriate for comparing models within this fixed protocol, not for claiming
that training results reproduce the original DrugCLIP training mixture or its
HomoAug augmentation.

## Fixed leakage and split policy

There is one policy in `protocol.json`:

- protein search: MMseqs2 `easy-search`, coverage `-c 0.8 --cov-mode 0`
  (alignment covers at least 80% of both query and target). `pident` is the
  identical aligned residue count divided by alignment length, as reported by
  MMseqs2. The nearest qualifying-coverage hit is recorded for every sample;
  identity **strictly greater than 30.0%** is excluded.
- train/dev clustering: MMseqs2 `easy-cluster --cluster-mode 2` at the same
  30% identity and bidirectional 80% coverage. Entire clusters are assigned by
  a fixed-seed shuffle until approximately 10% of complexes are in dev.
- ligand leakage: sanitized canonical isomeric SMILES + InChIKey exact match,
  and RDKit 2048-bit Morgan/ECFP4 (radius 2) against test **actives only**.
  Exact matches or Tanimoto **strictly greater than 0.60** are excluded.
- ProFSADB has no small-molecule filter. The released R/L complex lacks the
  full parent chain, so the pipeline resolves the PDB ID/chain encoded in sample
  paths against released source FASTA when present, then cached wwPDB SEQRES.
  Ambiguous/unresolved parents are excluded, never approximated with a pocket.

## One-command preparation (recommended)

The default materializer downloads
`zifeng-ai/target-based-drug-screening` at the verified immutable revision
`c8d2febc996e4c96b10f3aecd89fb7375150449f`, stages and validates the release, extracts the frozen
DUD-E/LIT-PCBA archives, and only then replaces the destination. The resolved
commit is recorded in `HF_MATERIALIZATION.json`. Configure the source through
`hf-source.json` or the `TARGET_BASED_DRUG_SCREENING_HF_REPO` and
`TARGET_BASED_DRUG_SCREENING_HF_REVISION` environment variables.

To regenerate from all pinned original sources instead, opt in explicitly:

```bash
TARGET_BASED_DRUG_SCREENING_DATA_SOURCE=upstream \
  bash environment/data/prepare.sh full /absolute/path/target-screening-data
```

Docker is the only host dependency. From the
`target-based-drug-screening/` directory, the
following command performs a preflight, builds the pinned materializer image,
runs a synthetic end-to-end smoke test, and then downloads, prepares, and
validates the full dataset:

```bash
bash environment/data/prepare.sh
```

By default, output is written to the git-ignored
`target-based-drug-screening/data/` directory. Set
`TARGET_BASED_DRUG_SCREENING_DATA_DIR` or pass an explicit target to store it
elsewhere:

```bash
bash environment/data/prepare.sh check /absolute/path/target-screening-data
bash environment/data/prepare.sh smoke /absolute/path/target-screening-data
bash environment/data/prepare.sh full /absolute/path/target-screening-data
```

On Windows, run this from WSL with Docker Desktop WSL integration enabled. A
path such as `D:\\datasets\\target-based-drug-screening` is available as
`/mnt/d/datasets/target-based-drug-screening` inside WSL. The compact Hub
release is about 15.8 GB before frozen test archive extraction. The
original-source route downloads about 60 GiB before extraction. The launcher
conservatively enforces 90 GiB of free space for either route. Re-running the
same command reuses a release already resolved to the pinned Hub commit; the
upstream route resumes `.part` downloads and complete preprocessing caches.
Progress is recorded in
`materialize.status`; success requires `.complete.json` plus a passing final
validation.

If Docker Desktop's WSL credential helper cannot read credentials while
pulling the public Python base image, the launcher automatically retries that
build with a temporary anonymous Docker CLI configuration. It does not modify
`~/.docker/config.json`.

Tune parallel structure preparation and retry behavior with
`TARGET_BASED_DRUG_SCREENING_WORKERS`,
`TARGET_BASED_DRUG_SCREENING_MAX_RETRIES`, and
`TARGET_BASED_DRUG_SCREENING_RETRY_DELAY_SECONDS`.

## Direct Python installation (advanced)

Python dependencies are pinned in `requirements.txt`; MMseqs2 is also required.
The supplied Materialize.Dockerfile installs both. This path is useful when Docker is not
available, but the one-command Docker launcher above is preferred.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# install mmseqs2 with the system/conda package manager
bash materialize.sh
```

The equivalent explicit form is:

```bash
TARGET_BASED_DRUG_SCREENING_DATA_DIR=/home/zifengw2/github/rsi-for-science/.agent-runtime/target-based-drug-screening/materialized \
  bash materialize.sh
```

For Docker, create the host directory once (it is already present in this
workspace), then run from `target-based-drug-screening/environment`:

```bash
bash run-data.sh
```

Compose mounts the same host `materialized/` directory at `/data`. Its
`create_host_path: false` setting prevents a misspelled path from silently
creating and populating a second dataset tree. It runs as host UID/GID
`1163739:1163739` by default so files written to this NFS workspace remain
owned by `zifengw2`. On another host, set
`TARGET_BASED_DRUG_SCREENING_UID=$(id -u)` and
`TARGET_BASED_DRUG_SCREENING_GID=$(id -g)` before invoking Compose.

On the Sunlab NFS host, `run-data.sh` also works around Docker root-squash: it
temporarily grants search-only (`o+x`) access on the user's home directory while
the container is running and restores the exact original mode on exit. This is
why the wrapper should be used instead of invoking `docker compose` directly.

BioLiP2 complex preparation runs with 24 sample-level worker processes by
default. Each worker keeps RDKit conformer generation single-threaded, uses the
same per-complex deterministic seed, and publishes NPZ/pickle cache entries by
atomic rename. Thus worker completion order cannot change dataset contents.
Set `TARGET_BASED_DRUG_SCREENING_WORKERS` to tune CPU/NFS pressure without
changing the protocol:

```bash
TARGET_BASED_DRUG_SCREENING_WORKERS=24 bash run-data.sh
```

Workers use an eight-entry receptor LRU by default and consume assembly-grouped
task chunks of 16 rows. These settings change only scheduling and may be tuned
without changing generated samples:

```bash
TARGET_BASED_DRUG_SCREENING_RECEPTOR_CACHE_SIZE=8 \
TARGET_BASED_DRUG_SCREENING_TASK_CHUNK_SIZE=16 \
bash run-data.sh
```

ProFSADB parsing, resumable Parquet shard publication, and selected-array
materialization use the same worker count. Completed shards and arrays are
reused after interruption.

Interrupted runs reuse every complete per-complex cache entry and recompute an
incomplete entry. Do not use `--overwrite` when resuming.

The pinned 2026-09-22 release has 11 explicitly documented stalled-tail
exclusions in `biolip2_manual_exclusions.json` (nine `kmer`, two `HEM`). They
are published as normal per-complex cache rows with
`exclusion_reason=manual_stalled_tail_exclusion`, so the canonical table keeps
all 560,724 unique source IDs. The heme entries are manual skips, not a claim
that heme is chemically invalid. Complete pre-existing cache entries are never
overwritten; a conflicting cached result fails closed. Rerunning preparation
resumes from all other complete cache entries and then builds the table.

The BioLiP2 site does not publish SHA256 values. The recipe pins byte sizes and
ETags, calculates SHA256 for every download, and writes
`raw/biolip2/download_manifest.json`. Freeze a first trusted run and enforce it
on later runs with:

```bash
bash materialize.sh --biolip-lock /trusted/download_manifest.json
```

Downloads use `.part` files and HTTP Range resume. Structural preprocessing is
cached per sample. Archive extraction and LMDB publication are idempotent or
atomic at file level. Expensive full stages can be rerun after interruption.

## Dry run

This runs the full parser → index → pocket → conformer → leakage → cluster →
split → canonical → LMDB → validation path on tiny synthetic fixtures. The
fixture supplies a command-compatible fake MMseqs executable only to exercise
orchestration; full mode refuses to proceed unless real `mmseqs` is installed.

```bash
bash materialize.sh /tmp/target-screening-data-smoke --smoke --overwrite
```

`SMOKE_DATA.json`, `DATASET_REPORT.md`, and `VALIDATION.json` all state that the
result is not training data. Full validation requires exactly 500,000 selected
ProFSADB IDs; smoke validation deliberately uses two.

## Output contract

The materialized root contains `raw/`, immutable `test/dude/` and
`test/lit_pcba/`, test Parquet/FASTA indices, canonical Parquet plus NPZ arrays,
fixed split text files, and three processed LMDB directories. LMDB is a derived
view; `canonical/*.parquet`, `canonical/arrays/`, and split files are the source
of truth. `DATASET_REPORT.md` records counts/distributions and
`VALIDATION.json` is written only on a passing build.

Individual stages are available through `scripts/*.py`; for example:

```bash
export TARGET_BASED_DRUG_SCREENING_DATA_DIR=/home/zifengw2/github/rsi-for-science/.agent-runtime/target-based-drug-screening/materialized
PYTHONPATH=. python scripts/download_test_data.py --data-root "$TARGET_BASED_DRUG_SCREENING_DATA_DIR"
PYTHONPATH=. python scripts/build_test_index.py --data-root "$TARGET_BASED_DRUG_SCREENING_DATA_DIR"
PYTHONPATH=. python scripts/validate_dataset.py --data-root "$TARGET_BASED_DRUG_SCREENING_DATA_DIR"
```
