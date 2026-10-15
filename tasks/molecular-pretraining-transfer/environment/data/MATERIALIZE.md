# Materializing the release (controller side)

Everything is rebuilt from public sources at pinned revisions (`build/SOURCES.sha256`). Nothing task-specific is
hosted. Run the pipeline in the task's runtime image, which has the pinned RDKit / LMDB / NumPy used to build it:

```bash
cd molecular-pretraining-transfer
docker build -f environment/Dockerfile --target runtime -t mpt-runtime .
docker run --rm -v "$PWD":/task:ro -v /large-disk/mpt:/out mpt-runtime \
  bash /task/environment/data/materialize.sh /out                 # add --verify-only to only check an existing release
export TASK_DATA_DIR=/large-disk/mpt TASK_PRIVATE_DIR=/large-disk/mpt
```

| step | script | output | time (208 cores) |
|---|---|---|---|
| download | `build/fetch.sh` | `sources/`: Uni-Mol2 shards 0–11 (179 GiB), Uni-Mol fine-tune data, COMPAS-1D, ExpansionRx, ASAP | network-bound |
| leakage filter | `build/filter_shards.py` | `public/pretrain/`: 12 training shards, `valid_part_0.lmdb`, `filter_report.json` (173 GiB) | about 16 min |
| downstream splits | `build/build_uni2_downstream.py` | QM9 and COMPAS-1D scaffold-similarity splits (temporary) | about 2 min |
| suite | `build/build_fast.py … full-test`, `build/build_admet.py` | 5k/1k fine-tune subsets, full test sets, ExpansionRx/ASAP conformers (temporary) | about 16 min |
| views | `build/make_views.py` | `public/downstream/`, `verifier/test-inputs/`, `private/labels/` | a minute |
| verify | `build/checksums.py verify` against `CHECKSUMS.json` | content digests of every released file | about 10 min cold, seconds from page cache |

Layout and who mounts what:

```text
OUT/public/pretrain/            development container (read-only)
OUT/public/downstream/          development container, verifier main (read-only)
OUT/verifier/test-inputs/       verifier main only: test molecules, labels replaced by 0.0
OUT/private/labels/             verifier grader only: test labels
OUT/sources/                    downloads; not mounted, may be deleted
```

The build is deterministic. LMDBs are compared by a digest of their (key, value) pairs, so the check does not
depend on page layout. The ASAP CSV is exported once from Polaris with `polaris-lib` (the only step that installs
a package); its sha256 is pinned like every other source.
